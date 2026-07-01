import socket
import json

code = """
import bpy
import json

# Take screenshot
filepath = r"D:\\ClothesNetData\\bpt\\validation_results\\blender_viewport.png"
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        with bpy.context.temp_override(area=area):
            bpy.ops.screen.screenshot_area(filepath=filepath)
        break

print("Screenshot saved: " + filepath)
"""

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(30)
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
        print('OK: ' + str(data.get('result', {}).get('result', '')))
    else:
        print('ERROR: ' + data.get('message', ''))
except Exception as e:
    print('Error: ' + str(e))
finally:
    sock.close()
