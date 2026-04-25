import urllib.request, json

req = urllib.request.Request('http://127.0.0.1:9191/reset', method='POST')
with urllib.request.urlopen(req) as r:
    d = json.loads(r.read())
    print('RESET - Scenario:', d.get('scenario'), '| max_steps:', d.get('max_steps'))

for i in range(4):
    req = urllib.request.Request('http://127.0.0.1:9191/run_agent_step', method='POST')
    with urllib.request.urlopen(req) as r:
        d = json.loads(r.read())
        print('STEP', i+1,
              '| steps_remaining:', d.get('steps_remaining'),
              '| det:', d.get('detection_level'),
              '| reward:', d.get('reward'),
              '| blue_tier:', d.get('blue_tier'),
              '| blocked:', d.get('blue_blocked'))
        if d.get('done'):
            print('Episode done!')
            break
