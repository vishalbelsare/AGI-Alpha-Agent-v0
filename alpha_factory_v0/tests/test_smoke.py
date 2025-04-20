import subprocess, os, json, time, tempfile, shutil, socket, requests, pathlib

def wait_port(host, port, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(1)
    raise TimeoutError(f"{host}:{port} not open")

def test_container_build_and_ui():
    tmp = tempfile.mkdtemp()
    shutil.copytree(".", tmp, dirs_exist_ok=True)
    img = "af:test"
    subprocess.run(["docker", "build", "-t", img, tmp], check=True)
    cid = subprocess.check_output(["docker", "run", "-d", "-p", "33000:3000", img]).decode().strip()
    try:
        wait_port("localhost", 33000)
        r = requests.get("http://localhost:33000/api/logs")
        assert r.status_code == 200
    finally:
        subprocess.run(["docker", "rm", "-f", cid])

