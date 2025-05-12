# part_b/tests/test_end_to_end.py

import sys
import os
import subprocess
import types
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# 1) Đưa root của repo vào sys.path để 'import part_a' và 'import part_b' được
# ──────────────────────────────────────────────────────────────────────────────
HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Patch pydeck (Streamlit) để khỏi văng KeyError khi import/deck
# ──────────────────────────────────────────────────────────────────────────────
_fake_pdk = types.SimpleNamespace(
    Layer=lambda *a, **k: None,
    Deck=lambda *a, **k: None,
    ViewState=lambda *a, **k: None
)
sys.modules['pydeck'] = _fake_pdk
sys.modules['part_b.gui.app.pdk'] = _fake_pdk

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
INTEGRATE = [sys.executable,
             os.path.join(REPO_ROOT, 'part_b', 'integrate.py')]

def run_cli(args):
    """Chạy integrate.py với list args, trả về CompletedProcess"""
    return subprocess.run(
        INTEGRATE + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# 3) Test cases CLI
# ──────────────────────────────────────────────────────────────────────────────

def test_cli_scats_only():
    # chỉ chạy Part A
    cp = run_cli(['--origin','2000','--dest','3002','--k','2','--mode','scats'])
    assert cp.returncode == 0
    out = cp.stdout
    assert "Part A: SCATS-based routing" in out
    # phải in ra 2 đường
    assert "Route 1:" in out and "Route 2:" in out
    # kết quả phải chứa site đích
    assert "3002" in out

def test_cli_osm_only():
    # chỉ chạy Part B
    cp = run_cli(['--origin','2000','--dest','3002','--k','2','--mode','osm'])
    assert cp.returncode == 0
    out = cp.stdout
    assert "Part B: OSM-based routing" in out
    assert "Route 1:" in out and "Route 2:" in out
    # kết quả phải chứa travel_time
    assert "travel_time" in out

def test_cli_both():
    # chạy cả 2 phần
    cp = run_cli(['--origin','2000','--dest','3002','--k','1','--mode','both'])
    assert cp.returncode == 0
    out = cp.stdout
    assert "Part A: SCATS-based routing" in out
    assert "Part B: OSM-based routing" in out
    # 1 tuyến cho mỗi phần
    assert out.count("Route 1:") == 2

@pytest.mark.parametrize("origin,dest", [
    ("9999","3002"),  # origin sai
    ("2000","9999"),  # dest sai
])
def test_cli_bad_site(origin,dest):
    cp = run_cli(['--origin',origin,'--dest',dest])
    assert cp.returncode != 0
    assert "origin/dest must be valid SCATS site IDs" in cp.stderr

def test_cli_bad_mode():
    cp = run_cli(['--origin','2000','--dest','3002','--mode','foobar'])
    assert cp.returncode != 0
    assert "invalid choice" in cp.stderr

@pytest.mark.parametrize("k", ["0","-1"])
def test_cli_bad_k(k):
    cp = run_cli(['--origin','2000','--dest','3002','--k',k])
    assert cp.returncode != 0
    assert "k must be > 0" in cp.stderr

def test_cli_missing_args():
    cp = run_cli([])
    assert cp.returncode != 0
    # thiếu origin/dest
    assert "usage:" in cp.stderr.lower()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Test Streamlit import (minimal) để đảm bảo GUI không crash ngay khi import
# ──────────────────────────────────────────────────────────────────────────────

def test_streamlit_app_import():
    # chỉ cần import module GUI và gọi hàm main() một lần
    import part_b.gui.app as gui_app
    # gọi nhẹ, không raise exception
    gui_app.main()
