from pathlib import Path
from collections import Counter
import json
import re
import xml.etree.ElementTree as ET
base = Path(__file__).resolve().parent
summary = {}
for path in sorted(base.glob('*.xml')):
    suite = ET.parse(path).getroot().find('testsuite')
    counts = {k: int(suite.get(k, 0)) for k in ['tests', 'failures', 'errors', 'skipped']}
    counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
    counts['seconds'] = float(suite.get('time'))
    log = path.with_suffix('.log').read_text()
    warnings = re.findall(r'(\d+) warnings?', log)
    counts['warnings'] = int(warnings[-1]) if warnings else 0
    counts['skip_reasons'] = dict(Counter(t.find('skipped').get('message') for t in suite.findall('testcase') if t.find('skipped') is not None))
    counts['failures_detail'] = [{'name': t.get('name'), 'type': child.tag, 'message': child.get('message')} for t in suite.findall('testcase') for child in t if child.tag in ['failure', 'error']]
    summary[path.stem] = counts
(base / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
for k, v in summary.items():
    print(k, {key: value for key, value in v.items() if key not in ['skip_reasons', 'failures_detail']})
    if v['failures_detail']: print(json.dumps(v['failures_detail'], indent=2))
