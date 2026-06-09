from Crypto.Cipher import AES
import os, json
import uuid

password = b'HFWCIewtpEPuGdICXPqlBndl936gO7z'
aes = AES.new(password, AES.MODE_ECB)

def parse_config():
    config = json.loads(open('config.json', 'r', encoding='utf-8').read())
    return config

def encrypt(input_str):
    # 补充到16的倍数
    input_byte = input_str.encode()
    input_byte = (input_byte + b'\x00'*16)[:16 * ((len(input_byte)//16) + 1)]
    return aes.encrypt(input_byte)

def decrypt(input_byte):
    decrypt_byte = aes.decrypt(input_byte)
    decrypt_byte = decrypt_byte[:-16] + decrypt_byte[-16:].replace(b'\x00', b'')
    return decrypt_byte.decode()

def encrypt_imgs64(input_str, save_type, process_num, video_num, msg_id):
    config = parse_config()
    if config['imgsave_turn'] != 'on':
        return None
    save_dir = config['imgsave_register_dir'] if save_type == 'register' else config['imgsave_detection_dir']

    if not os.path.exists(f'{save_dir}/{process_num}/{video_num}'):
        os.makedirs(f'{save_dir}/{process_num}/{video_num}')

    uuid_str = str(uuid.uuid4())
    save_path = f'{save_dir}/{process_num}/{video_num}/{msg_id}_{uuid_str}.jpg'.replace('//', '')
    encrypt_byte = encrypt(input_str)
    with open(save_path, 'wb') as f:
        f.write(encrypt_byte)
    return None
