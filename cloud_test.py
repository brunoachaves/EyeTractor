from azure.iot.device import IoTHubDeviceClient, Message
from datetime import datetime


# Metodo de monitoramento
# eventType pode ser 'normal' ou 'desatencao'
def send_message(eventType):
    CAMERAID = "camera-01"
    TODAY = datetime.now().isoformat()

    msg_txt_formatted = MSG_TXT.format(TODAY=TODAY, EVENT=eventType, CAMERAID=CAMERAID)
    message = Message(msg_txt_formatted)

    print("Enviando mensagem: {}".format(message))
    try:
        client.send_message(message)
        print(f"[Info] Mensagem {message} enviada com sucesso")
    except:
        print("[Erro] Falha ao enviar a mensagem")


# Cria instancia do client do iot hub
CONNECTION_STRING = "HostName=eyetractor-hubiot.azure-devices.net;DeviceId=camera-01;SharedAccessKey=5pwN7TTPM7/V0bqXI+wEruGdCJi1o3h6QLEQRNwua+g="
client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
# Mensagem a ser enviada para o HubIot
MSG_TXT = '{{"CameraId": "{CAMERAID}", "EventType": "{EVENT}", "EventDate": "{TODAY}" }}'

qtd_drowsiness = 0
qtd_distraction = 0


if __name__ == '__main__':
    while True:
        op = input('Enviar sinal de fadiga <0> ou distração <1>?')
        if op == '0':
            send_message("fadiga")
            qtd_drowsiness += 1
        else:
            send_message("distracao")
            qtd_distraction += 1
        print()
        print(f'Total de fadiga: {qtd_drowsiness}')
        print(f'Total de distração: {qtd_distraction}')
        print()
