"""List all runs in mluo/deepscaler-1.5b with summary metrics, pick shortest."""
import json
import urllib.request

KEY = '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'
URL = 'https://api.wandb.ai/graphql'
ENTITY = 'mluo'
PROJECT = 'deepscaler-1.5b'


def gql(query, variables=None, timeout=120):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}",
                 "User-Agent": "wandb-shortest-length-solve"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()


LIST_RUNS = """
query ListRuns($entity: String!, $project: String!, $cursor: String, $limit: Int) {
  project(name: $project, entityName: $entity) {
    name
    runCount
    runs(first: $limit, after: $cursor) {
      edges {
        node {
          name
          displayName
          state
          createdAt
          summaryMetrics
          historyKeys
        }
      }
      pageInfo { endCursor hasNextPage }
    }
  }
}
"""


def parse_summary(sm_raw):
    try:
        sm = json.loads(sm_raw) if sm_raw else {}
    except Exception:
        sm = {}
    return sm


def find_response_mean(sm):
    # try several key forms
    for k in ['response_length/mean', 'response_length_mean', 'response_length']:
        if k in sm:
            v = sm[k]
            if isinstance(v, dict):
                return v.get('value', v.get('mean'))
            return v
    # nested
    for k, v in sm.items():
        if isinstance(v, dict) and 'response_length' in json.dumps(k).lower():
            pass
    return None


def main():
    runs = []
    cursor = None
    while True:
        vars_ = {"entity": ENTITY, "project": PROJECT, "limit": 50, "cursor": cursor}
        s, b = gql(LIST_RUNS, vars_)
        data = json.loads(b)
        if 'errors' in data:
            print('ERRORS:', json.dumps(data['errors'])[:1000])
            break
        proj = data['data']['project']
        edges = proj['runs']['edges']
        for e in edges:
            n = e['node']
            sm = parse_summary(n.get('summaryMetrics'))
            runs.append({
                'name': n['name'],
                'displayName': n.get('displayName'),
                'state': n.get('state'),
                'createdAt': n.get('createdAt'),
                'summary': sm,
                'historyKeys': n.get('historyKeys'),
            })
        pi = proj['runs']['pageInfo']
        if not pi.get('hasNextPage'):
            break
        cursor = pi['endCursor']
    print('total runs:', len(runs))
    with open('runs_list.json', 'w') as f:
        json.dump(runs, f, indent=1)
    # print summary of interest
    for r in runs:
        sm = r['summary']
        rl = find_response_mean(sm)
        print(r['name'], r.get('displayName'), r['state'], 'resp_mean=', rl,
              'keys_sample=', [k for k in sm.keys() if 'response' in k.lower() or 'entropy' in k.lower() or 'clip' in k.lower()][:10])


if __name__ == '__main__':
    main()
