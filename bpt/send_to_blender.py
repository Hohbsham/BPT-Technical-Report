import socket
import json

script_path = r'D:\ClothesNetData\bpt\visualize_in_blender.py'
with open(script_path, 'r') as f:
    code = f.read()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(120)
try:
    sock.connect(('localhost', 9876))

    msg = json.dumps({'type': 'execute_code', 'params': {'code': code}})
    sock.sendall(msg.encode('utf-8'))

    chunks = []
    while True:
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
        except socket.timeout:
            break

    response = b''.join(chunks).decode('utf-8')
    data = json.loads(response)
    if data.get('status') == 'success':
        print('SUCCESS: Script executed in Blender')
        result = data.get('result', {})
        if isinstance(result, dict) and result.get('executed'):
            stdout = result.get('result', '')
            if stdout:
                print(stdout[:3000])
    else:
        print('ERROR: ' + data.get('message', 'Unknown error'))
except Exception as e:
    print('Connection error: ' + str(e))
finally:
    sock.close()
