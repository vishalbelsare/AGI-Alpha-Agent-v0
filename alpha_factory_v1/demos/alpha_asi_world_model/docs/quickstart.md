## Quick‑Start

```bash
pip install -r requirements.txt
python -m alpha_asi_world_model_demo --demo
```

To run without internet access, build wheels on a connected machine and
install from a local wheelhouse:

```bash
mkdir -p /media/wheels
pip wheel -r requirements.txt -w /media/wheels
pip wheel -r ../../../requirements-dev.txt -w /media/wheels
WHEELHOUSE=/media/wheels AUTO_INSTALL_MISSING=1 \
  python check_env.py --auto-install --wheelhouse /media/wheels
WHEELHOUSE=/media/wheels alpha-asi-demo --demo
```