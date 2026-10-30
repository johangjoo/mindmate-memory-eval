"""Validate published evidence offline. Does not call APIs or connect to a database."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8),(a,b)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tokens',action='store_true');args=parser.parse_args()
    sessions=read('data/sessions.json');questions=read('data/questions_gold.json')
    events=read('data/events_gold.json');summary=read('results/summary.json')
    final=read('results/final_answers.json');changes=read('results/adjudications.json');usage=read('results/api_usage.json')
    assert len(sessions)==10 and len(events)==30 and len(questions)==95
    assert len({q['question_id'] for q in questions})==95
    assert Counter(q['phase'] for q in questions)==dict(checkpoint=66,in_session=5,final_flush=24)
    turns={t['turn_id']:(s,t,i) for s in sessions for i,t in enumerate(s['turns'])}
    assert len(turns)==sum(len(s['turns']) for s in sessions)==206
    assert all(s['synthetic'] is True for s in sessions)
    for q in questions:
        assert q['save_to_memory'] is False and q['independent_request'] is True
        assert q['as_of_date']==sessions[q['after_session']-1]['session_date']
        for tid in q['evidence_turn_ids']:
            s,t,i=turns[tid]
            assert s['session_index']<=q['after_session'],q['question_id']
            if q.get('after_turn_id') and s['session_index']==q['after_session']:
                assert i<=turns[q['after_turn_id']][2],q['question_id']
    for event in events:
        assert all(tid in turns for tid in event['source_turn_ids'])
    assert len(summary)==6 and len(final)==144
    qmap={q['question_id']:q for q in questions}
    overrides={(r['condition'],r['record_id']):r for r in changes}
    for m in summary:
        condition=m['condition'];rows=read('results/answers/'+condition+'.json')
        finals=[r for r in rows if r['phase']=='final_flush']
        checkpoints=[r for r in rows if r['phase']=='checkpoint']
        assert len(finals)==24 and len(checkpoints)==66
        assert {r['question_id'] for r in finals}=={q['question_id'] for q in questions if q['phase']=='final_flush'}
        assert finals==[r for r in final if r['condition']==condition]
        assert sum(r['automatic_score'] for r in finals)==m['final_automatic']
        assert sum(r['reported_score'] for r in finals)==m['final_reported']
        assert sum(r['automatic_score'] for r in checkpoints)==m['checkpoint_automatic']
        assert sum(r['reported_score'] for r in checkpoints)==m['checkpoint_reported']
        for r in rows:
            assert r['automatic_score'] in [0,1] and r['reported_score'] in [0,1]
            assert r['expected_answer']==qmap[r['question_id']]['expected_answer']
            edit=overrides.get((condition,r['record_id']))
            if edit:
                assert edit['automatic']==r['automatic_score'] and edit['reported']==r['reported_score']
                assert edit['reason']==r['review_adjustment_reason']
            else:
                assert r['reported_score']==r['automatic_score'] and not r['review_adjustment_reason']
        calls=[r for r in usage if r['condition']==condition]
        good=[r for r in calls if r['successful_storage']]
        bad=[r for r in calls if r['purpose'] in ['extraction','reconciliation','embedding'] and not r['successful_storage']]
        evaluation=[r for r in calls if r['purpose'] in ['reader','judge']]
        assert sum(r['input_tokens'] for r in good)==m['storage_input_tokens']
        assert sum(r['output_tokens'] for r in good)==m['storage_output_tokens']
        assert sum(r['cached_tokens'] for r in good)==m['cached_input_subset']
        assert sum(r['reasoning_tokens'] for r in good)==m['reasoning_output_subset']
        close(sum(r['usd'] for r in good),m['storage_usd'])
        close(sum(r['usd'] for r in bad),m['failed_storage_usd'])
        close(sum(r['usd'] for r in evaluation),m['evaluation_usd'])
        close(sum(r['usd'] for r in calls),m['actual_api_usd'])
        close(m['storage_usd']+m['start_usd_estimate'],m['operating_usd_estimate'])
        close(m['start_input_tokens_estimate']*4/1_000_000,m['start_usd_estimate'])
        assert m['storage_plus_start_tokens']==m['storage_input_tokens']+m['storage_output_tokens']+m['start_input_tokens_estimate']
    with (ROOT/'results/final_answers.csv').open(encoding='utf-8-sig',newline='') as f:
        csvrows=list(csv.DictReader(f))
    assert len(csvrows)==144
    for a,b in zip(csvrows,final):
        for key in ['condition','record_id','question_id','answer','expected_answer','review_adjustment_reason']:
            assert a[key]==b[key]
        assert int(a['reported_score'])==b['reported_score']
    for source in read('provenance/source_manifest.json')['sources']:
        assert hashlib.sha256((ROOT/source['published_path']).read_bytes()).hexdigest()==source['sha256']
    checksum_path=ROOT/'provenance/SHA256SUMS.json'
    if checksum_path.exists():
        checks=read('provenance/SHA256SUMS.json')
        for path,sha in checks.items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    if args.tokens:
        import tiktoken
        enc=tiktoken.get_encoding('o200k_base')
        tokens=sum(len(enc.encode('\n'.join(t['role']+': '+t['text'] for t in s['turns']))) for s in sessions)
        assert tokens==9560,tokens
        print('Raw dialogue: 9,560 o200k_base tokens')
    print('PASS: 10 sessions / 206 turns / 30 events / 95 questions / 144 final answers / 6 conditions')
    print('PASS: source links, temporal evidence boundaries, adjudications, scores, token/cost totals, CSV and hashes')

if __name__=='__main__':main()
