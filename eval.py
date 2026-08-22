import json
import os
import time
from pathlib import Path

# sys.path.append(os.environ.get("EVAL_INFER_PATH"))

from infer import infer


if __name__ == '__main__':
    required_env = ('EVAL_DATA_PATH', 'MODEL_OUTPUT_PATH')
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    os.environ.setdefault('EVAL_RESULT_PATH', './outputs/eval')

    os.makedirs(os.environ.get('EVAL_RESULT_PATH'), exist_ok=True)


    result = {}

    t0 = time.time()
    top10s, user_list = infer()
    t1 = time.time()

    result['time'] = t1 - t0
    result['top10s'] = top10s
    result['user'] = user_list
    

    retrieved_less_10 = sum(1 for x in top10s if len(x) < 10)
    if retrieved_less_10 > 0:
        print(f'Warning: {retrieved_less_10 / len(top10s):.3f} test samples matched less than 10 results')

    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "result.json"), 'w') as f:
        json.dump(result, f)
