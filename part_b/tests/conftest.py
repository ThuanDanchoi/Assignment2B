# part_b/tests/conftest.py
import sys, os

# Đưa root của repo (lên hai cấp) vào đầu sys.path
ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
