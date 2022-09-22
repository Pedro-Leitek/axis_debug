import sys,os
import time
import uuid
import math
import argparse
import axis
import asyncio
from httpx import AsyncClient
from axis.ptz import PtzControl
from urllib.parse import urlparse
import async_timeout
HOST = '192.168.180.7'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
HOST2 = '192.168.16.7'  # Standard loopback interface address (localhost)
PORT2 = 65433
#HOST3 = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT3 = 65434
PORT4 = 65435
UTILIZADORES_ATIVOS = {}  # Lista de mantem todos os utilizadores em processo no servidor
sys.path.insert(1, os.path.join(sys.path[0], ".."))

# Obtem dados do utilizador, usado para obter url da camara e thumbnail, obter objeto da camara e video
def obtemCrm(user, vid):
    if user in UTILIZADORES_ATIVOS and vid in UTILIZADORES_ATIVOS.get(user).camaras:
        return UTILIZADORES_ATIVOS.get(user).camaras.get(vid)

def obtemPTZ(user, vid):
    if user in UTILIZADORES_ATIVOS and vid in UTILIZADORES_ATIVOS.get(user).camaras:
        return UTILIZADORES_ATIVOS.get(user).ptz.get(vid)

def obtemVideo(user, vid):
    if user in UTILIZADORES_ATIVOS and vid in UTILIZADORES_ATIVOS.get(user).videos:
        return UTILIZADORES_ATIVOS.get(user).videos.get(vid)


def ObtemExistenciaCmr(nome, nomeCmr):
    v = UTILIZADORES_ATIVOS.get(nome).camaras.values()
    for i in v:
        if nomeCmr == i.nome:
            return True
    return False

def ApagaCamara(user, vid):
    if user in UTILIZADORES_ATIVOS and vid in UTILIZADORES_ATIVOS.get(user).camaras:
        UTILIZADORES_ATIVOS.get(user).camaras.get(vid).imagem.release()
        UTILIZADORES_ATIVOS.get(user).camaras.get(vid).thread.join()
        d = UTILIZADORES_ATIVOS.get(user)
        del d.camaras[vid]

class AIEngine:
    def __init__(self):
        #self.id = id_u
        self.camaras = {}  # vid: objeto
        self.videos = {}  # vid: objeto
        self.ptz = {}

    def CriaCamara(self, lnk, nome):#filtros
        #vid = str(uuid.uuid1()).replace("-", "")  # Gera o id do video que também é usado para url para aceder via web
        cmr = Camara(lnk, nome)
        #videredb.inserirStream(nome, vid,"manual")
        return cmr

    def CriaPTZ(self, lnk, nome):
        p= PTZ(lnk,1)
        return p   
        
    def obtemCamarasLigacao(self):
        cmrs = {}
        for i in self.camaras:
            cmrs[self.camaras[i].nome] = i
        return cmrs


def move(p,angle,direction):
    if direction=="Left":
        p.move_left(p.loop, angle)
        time.sleep(0.8)
        get_pos(p)
    if direction=="Right":
        p.move_right(p.loop, angle)
        time.sleep(0.8)
        get_pos(p)
    if direction=="Up":
        p.move_up(p.loop, angle)
        time.sleep(0.8)
        get_pos(p)
    if direction=="Down":
        p.move_down(p.loop, angle)
        time.sleep(0.8)
        get_pos(p)

def go(p,azimuth,elevation):
    go_azi(p,azimuth)
    time.sleep(0.8)  
    go_ele(p,elevation)
    time.sleep(0.8)
    get_pos(p)

def go_azi(p,azimuth):     
    p.move_go_azi(p.loop, azimuth)

def go_ele(p,elevation):
    p.move_go_ele(p.loop, elevation)

def init_azi(p):
    p.init_position_azi(p.loop)

def init_ele(p):
    p.init_position_ele(p.loop)

def zoom_in_out(p,zoom):
    p.zoom=int(zoom)
    p.zoom_func(p.loop, zoom)

def init_zoom_in_out(p):
    p.zoom=1
    p.init_zoom(p.loop)

def start_speeddry(p):
    p.speeddry(p.loop)

def set_ircutfilter(p, ir):
    p.ircutfilter(p.loop,ir)
    

def get_pos(p):
    pos=p.get_position(p.loop)
    x=pos.split("\n")
    pan=x[0].split("pan=")
    tilt=x[1].split("tilt=")
    if float(pan[1])<0.0:
       p.azimuth=360+float(pan[1])
    else:
       p.azimuth=float(pan[1])
    p.elevation=float(tilt[1])
    

class PTZ:
    def __init__(self,lnk,user):
         self.url = urlparse(lnk)
         self.username=self.url.username
         self.password=self.url.password
         self.host=self.url.hostname
         self.loop = asyncio.new_event_loop()
         self.p=self.create_ptz(self.loop)
         self.user=user
         print(self.username)
         print(self.password)
         print(self.host)
         self.angle=0.0
         self.azimuth=0.0
         self.elevation=0.0
         self.zoom=1
         self.direction=""
         #thr1=threading.Thread(target=tour,args=(self.user, self.vid))
         #thr1.start()

    async def aux_left(self,p,angle):
         #if float(angle)>=360.0:
         #   angle=360.0
         #elif float(angle)<0.0:
         #   angle=0.0
         #if self.azimuth - float(angle) < 0.0:
         #   self.azimuth = 360.0 + self.azimuth - float(angle)
         #else:
         await p.control(camera=1, rpan=-float(angle))

    async def aux_right(self,p,angle):
         #if float(angle)>=360.0:
         #   angle=360.0
         #elif float(angle)<0.0:
         #   angle=0.0
         #if self.azimuth + float(angle) >= 360.0:
         #   self.azimuth = -360.0 + self.azimuth + float(angle)
         #else:
         #   self.azimuth += float(angle)
         await p.control(camera=1, rpan=float(angle))

    async def aux_up(self,p,angle):
         #if self.elevation + float(angle) > 20.0:
          #  self.elevation = 20.0
         #else:
          #  self.elevation += float(angle)
         await p.control(camera=1, rtilt=float(angle))

    async def aux_down(self,p,angle):
         #if self.elevation - float(angle) < -90.0:
          #  self.elevation = -90.0
         #else:
         #   self.elevation += -float(angle)
         await p.control(camera=1, rtilt=-float(angle))

    async def aux_go_azi(self,p,azimuth):
         if float(azimuth) > 180.0:
            azi=-360.0 + float(azimuth)
         else:
            azi=float(azimuth)
         #self.azimuth = float(azimuth)
         await p.control(camera=1, pan=azi)

    async def aux_go_ele(self,p,elevation):
         #self.elevation = float(elevation)
         await p.control(camera=1, tilt=float(elevation))

    async def aux_init_position_azi(self,p):
         #self.azimuth=0.0
         await p.control(camera=1, pan=0)

    async def aux_init_position_ele(self,p):
         #self.elevation=0.0
         await p.control(camera=1, tilt=0)

    async def aux_init_zoom(self,p):
         #self.zoom=1
         await p.control(camera=1, zoom=1)

    async def aux_zoom(self,p,zoom):
         #self.zoom=int(zoom)
         await p.control(camera=1, zoom=int(zoom))

    async def aux_get_position(self,p):
         return await p.query("position")
    
    async def aux_speeddry(self,p):
         print("speeddry")
         await p.control(camera=1, auxiliary="speeddry")

    async def aux_ircutfilter(self,p, ir):
         await p.control(camera=1, ircutfilter=ir)

    def move_left(self,loop,angle):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_left(self.p,angle))

    def move_right(self,loop,angle):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_right(self.p,angle))

    def move_up(self,loop,angle):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_up(self.p,angle))

    def move_down(self,loop,angle):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_down(self.p,angle))

    def move_go_azi(self,loop,azimuth):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_go_azi(self.p,azimuth))

    def move_go_ele(self,loop,elevation):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_go_ele(self.p,elevation))

    def init_position_azi(self,loop):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_init_position_azi(self.p))

    def init_position_ele(self,loop):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_init_position_ele(self.p))

    def zoom_func(self,loop,zoom):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_zoom(self.p,zoom))

    def init_zoom(self,loop):
         asyncio.set_event_loop(loop)  
         loop.run_until_complete(self.aux_init_zoom(self.p))

    def get_position(self,loop):
         asyncio.set_event_loop(loop)  
         return loop.run_until_complete(self.aux_get_position(self.p))

    def speeddry(self,loop):
         asyncio.set_event_loop(loop)
         loop.run_until_complete(self.aux_speeddry(self.p))

    def ircutfilter(self,loop, ir):
         asyncio.set_event_loop(loop)
         loop.run_until_complete(self.aux_ircutfilter(self.p, ir))            

    async def device_creation(self):
        device = await self.axis_device(self.host, 80, self.username, self.password)
        print(3333333)
        p = self.ptz_control(device)
        return p

    def create_ptz(self,loop):
        #loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        aux=loop.run_until_complete(self.device_creation())
        #loop.close()
        return aux

    def ptz_control(self,axis_device) -> PtzControl:
        """Returns the PTZ control mock object."""
        print(11111111)
        print(axis_device.vapix.request)
        return PtzControl(axis_device.vapix.request)

    async def axis_device(self,
    host: str, port: int, username: str, password: str
    ) -> axis.AxisDevice:
        """Create a Axis device."""
        session = AsyncClient(verify=False)
        device = axis.AxisDevice(
            axis.configuration.Configuration(
                session, host, port=port, username=username, password=password
            )
        )
        try:
            with async_timeout.timeout(5):
                await device.vapix.initialize_users()
                await device.vapix.load_user_groups()
            await device.vapix.initialize_event_instances()

            return device

        except axis.Unauthorized:
            LOGGER.warning(
            "Connected to device at %s but not registered or user not admin.", host
        )

        except (asyncio.TimeoutError, axis.RequestError):
            LOGGER.error("Error connecting to the Axis device at %s", host)

        except axis.AxisException:
            LOGGER.exception("Unknown Axis communication error occurred")

        return device
    
    def getAzimuth(self):
        connected2=False
        s2=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while not connected2:
           #print("tttttttttt")  
            # attempt to reconnect, otherwise sleep for 2 seconds  
           try:  
              s2.connect((HOST2, PORT4))
              s2.sendall(json.dumps({"id":1}).encode('utf-8'))
              accept2=s2.recv(1024).decode('ascii')
              if accept2=='True':
                 connected2=True
              else:
                 s2.close()
                 s2=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                 time.sleep( 2 )
           except socket.error:  
              time.sleep( 2 )

        while True:
              #print(self.azimuth)
              info = json.dumps({"azimuth":self.azimuth}).encode('utf-8')
              try:
                 s2.sendall(len(info).to_bytes(8,'big'))
                #print("Before ack")
                 ack=s2.recv(1024).decode('ascii')
                #print(len(frame))
                 if ack=="ACK":
                    s2.sendall(info)
                    ack=""
                    #self.framecurrente=None
                    #utilizador2.UTILIZADORES_ATIVOS[main.session["user_id"]].CriaCamara(linkCamara, nomeCamara) #filtroObjetos)
                 time.sleep( 1 )
              except socket.error:
                 connected2 = False
                 s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                 while not connected2:  
            # attempt to reconnect, otherwise sleep for 2 seconds  
                    try:  
                       s2.connect((HOST2, PORT4))
                       s2.sendall(json.dumps({"id":1}).encode('utf-8'))
                       accept2=s2.recv(1024).decode('ascii')
                       if accept2=='True':
                          connected2=True
                       else:
                          s2.close()
                          s2=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                          time.sleep( 2 )
                    except socket.error:  
                          time.sleep( 2 ) 
              
        
s1=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
class Camara:
    def __init__(self, lnk, c_nome):#filtros
        #self.filtros = filtros
        self.nome = c_nome
        #self.id = vid
        #self.id_user = id_user
        #self.link = cv2.VideoCapture(lnk, 0)
        self.lnk = lnk; 
        self.net = cv2.dnn.readNetFromONNX(config.yoloPathWeights)
        self.framecurrente = None
        self.tempoInicial = time.time()  # Tempo inicial da contagem para guardar a proxima frame da BD
        self.tempoPassado = 0
        self.thread = None
        self.url = urlparse(lnk)
        self.retoques = True
        self.brilho = 0
        self.contraste = 0
        self.mode="auto"
        self.tour=[]
        print(self.url.hostname)
        print(cv2.cuda.getCudaEnabledDeviceCount())
        # Inicia CUDA, se utilizador não suportar, estas linhas são ignoradas
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def obtemFrame(self, p):
        connected2=False
        s2=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while not connected2:  
            # attempt to reconnect, otherwise sleep for 2 seconds  
           try:  
              s2.connect((HOST2, PORT3))
              s2.sendall(json.dumps({"id":1}).encode('utf-8'))
              accept2=s2.recv(1024).decode('ascii')
              if accept2=='True':
                 connected2=True
              else:
                 s2.close()
                 s2=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                 time.sleep( 2 )
           except socket.error:  
              time.sleep( 2 )
        #s2.setblocking(0)
        while True:
            #print("rtrttrttrtrtrtrtr")
            if self.framecurrente is None:
               continue
               '''img = cv2.imread(
                    "static\img\espera.png")  # Caso o frame não esteja disponivel então mete uma imagem de espera
                _, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n' '''
            else:
                #frame = self.framecurrente.tolist()
                _, buffer = cv2.imencode('.jpg', self.framecurrente)
                frame = buffer.tobytes()
                #info = json.dumps({"frame":frame,"azimuth":p.azimuth}).encode('utf-8')
                #print("rtrttrttrtrtrtrtr")
                #frame = self.framecurrente
                #print(len(frame))
            #yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                try:
                   s2.sendall(len(frame).to_bytes(8,'big'))
                   print("Before ack")
                   #ready = select.select([s2], [], [], 2)
                   #if ready[0]:
                     #ack=s2.recv(1024).decode('ascii')
                     #print(ack)
                     #if ack=="ACK":
                   s2.sendall(frame)
                       #ack=""
                   self.framecurrente=None
                   s2.recv(1024).decode('ascii')
                   #time.sleep( 1 )
                    #utilizador2.UTILIZADORES_ATIVOS[main.session["user_id"]].CriaCamara(linkCamara, nomeCamara) #filtroObjetos)
                except socket.error:
                   connected2 = False
                   s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                   while not connected2:  
            # attempt to reconnect, otherwise sleep for 2 seconds  
                      print("reconnection")
                      try:  
                         s2.connect((HOST2, PORT3))
                         s2.sendall(json.dumps({"id":1}).encode('utf-8'))
                         accept2=s2.recv(1024).decode('ascii')
                         if accept2=='True':
                            connected2=True
                         else:
                            s2.close()
                            s2=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            time.sleep( 2 )
                      except socket.error:  
                            time.sleep( 2 ) 

    def obtemThumbnail(self):
        if self.framecurrente is None:
            return None
        else:
            _, buffer = cv2.imencode('.jpg', self.framecurrente)
            frame = buffer.tobytes()
            return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

    def __del__(self):
        self.imagem.release()

    def stream(self,p):
        print("stream")
        imagem=cv2.VideoCapture(self.lnk, 0)
        init_azi(p)
        time.sleep(0.8)
        init_ele(p)
        time.sleep(0.8)
        init_zoom_in_out(p)
        time.sleep(0.8)
        get_pos(p)   
        while True: 
          if self.mode=="manual":         
            ativo, frame = imagem.read()
            #img = np.int16(frame)
            #img = img * (self.contraste / 127 + 1) - self.contraste + self.brilho
            #img = np.clip(img, 0, 255)
            #frame = np.uint8(img)
            cv2.rectangle(frame, (0,0), (frame.shape[1],60), (0,0,0), -1)
            cv2.putText(frame, "Leitek-Demo1" + " " + "Time:" + " " + time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(time.time())) +  " " + "Azimuth:" +  " " + 					str(p.azimuth) + " " + "Elevation:"+ " " + str(p.elevation), (0, 45), 		 cv2.FONT_HERSHEY_DUPLEX, 1.25, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            self.framecurrente=frame
            #yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
          else:
            zoom_in_out(p,2250) #2250
            time.sleep(0.8)
            i=0
            j=0
            while self.mode=="auto":
               print("Postprocessing")
               #self.framecurrente=frame
               print("Aqui!!!!!")
               go_azi(p,15.0*i)
               print("Aqui!!")
               time.sleep(0.8)
               if i==0:
                  go_ele(p,-9.0*j)
                  time.sleep(0.8)
                  j=j+1
                  if j>2:
                    j=0                 
               get_pos(p)
               time.sleep(0.8)
               i=i+1
               if i>23:
                 i=0
                  #go_ele(p,-9.0)
                  #time.sleep(0.8)
                  #get_pos(p)
                  #time.sleep(0.8)
                  #self.processa(p)




            '''init_azi(p)
            time.sleep(0.8)
            init_ele(p)
            time.sleep(0.8)
            zoom_in_out(p,2250) #2250
            time.sleep(0.8)
            get_pos(p)
            i=1
            j=1
            self.framecurrente=None 
            while self.mode=="auto":
               self.processa(p)
               print("Postprocessing")
               #self.framecurrente=frame
               print("Aqui!!!!!")
               go_azi(p,15.0*i)
               print("Aqui!!")
               time.sleep(1)
               if i==0:
                  go_ele(p,-9.0*j)
                  time.sleep(0.8)
                  j=j+1
                  if j>2:
                    j=0                 
               get_pos(p)
               i=i+1
               if i>23:
                 i=0'''
                   
                 
            #imagem=cv2.VideoCapture(self.lnk, 0)

    def processa(self,ptz):
        imagem=cv2.VideoCapture(self.lnk, 0)
        if imagem.isOpened():
            print("Camara is connected")
            ativo, frame = imagem.read()
            #if not ativo: break
            #frame = cv2.imread("Screenshot from 2022-01-19 17-35-17_180.png")
            # Ajusta imagem
            #img = np.int16(frame)
            #img = img * (self.contraste / 127 + 1) - self.contraste + self.brilho
            #img = np.clip(img, 0, 255)
            #frame = np.uint8(img)
            tamanho = frame.shape
            #print(tamanho)
            layer_names = self.net.getLayerNames()
            #outputlayers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            outputlayers = [layer_names[i-1] for i in self.net.getUnconnectedOutLayers()] 
            altura = frame.shape[0]
            comprimento = frame.shape[1]
            print("Preprocessing1")
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
            print("Postprocessing1")
            print("Preprocessing2")
            self.net.setInput(blob)
            print("Postprocessing2")
            print("Preprocessing3")
            outputs = self.net.forward(outputlayers)
            print("Postprocessing3")
            #print("Inicio processamento")
            class_ids = []
            confidences = []  # Grau de confiança sobre a imagem
            caixas = []
            rows = outputs[0].shape[1]
            x_factor = 1920 / 640
            y_factor =  1080 / 640


            for r in range(rows):

               row = outputs[0][0][r]

               confidence = row[4]

            # Discard bad detections and continue.
               if confidence >= 0.5:

                    classes_scores = row[5:]
                    class_id = np.argmax(classes_scores)

                    '''for p in processados:
                    for ObjetoApanhado in p:
                    pontuacoes = ObjetoApanhado[5:]
                    class_id = np.argmax(pontuacoes)

                    #if class_id not in self.filtros:  # Este objeto detetado não está presente nos filtros para detetar
                        #continue

                    certeza = pontuacoes[class_id]'''
                    if classes_scores[class_id] > 0.5: 
                    #if certeza > 0.5:
                        # Obtem posição (eu não percebo a magia negra que o net.forward faz, mas as posições das coisas estão ai)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]

                        left = int((cx - w/2) * x_factor)

                        top = int((cy - h/2) * y_factor)

                        width = int(w * x_factor)

                        height = int(h * y_factor)

                        '''centroX = int(row[0] * comprimento)
                        centroY = int(row[1] * altura)
                        c = int(row[2] * comprimento)
                        a = int(row[3] * altura)
                        x = int(centroX - c / 2)
                        y = int(centroY - a / 2)'''

                        caixas.append([left, top, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            print("Postprocessing4")    
            indexes = cv2.dnn.NMSBoxes(caixas, confidences, 0.5, 0.4)
            objetos_capturados_frame = []
            scores = []
            for i in indexes:
                objeto_no_frame = {}
                #i = i[0]
                caixa = caixas[i]
                x = caixa[0]
                y = caixa[1]
                w = caixa[2]
                h = caixa[3]
                objeto_no_frame["object_id"] = int(class_ids[i])
                objeto_no_frame["confianca"] = round(confidences[i],2)
                objeto_no_frame["topLeft"] = [x, y]
                objeto_no_frame["bottomRight"] = [w, h]
                objetos_capturados_frame.append(objeto_no_frame)
                label = str(dataset.classes[class_ids[i]])
                cor = dataset.classes_cores[class_ids[i]]
                #cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
                scores.append(round(confidences[i],2))
                # Impede que texto fique fora do ecrã
                #if y < 40:  # se o X estiver 40 pixeis perto do topo da imagem
                #    y += 45  # Baixa 45 pixeis
                #if x < 5: x += 5

                #if x + (len(label) + 5) * 15 > tamanho[1]:  # Impede que o texto saia para o lado da imagem
                #    x -= (len(label) + 5) * 15

                # Faz um texto em baixo que serve como contorno
                #cv2.putText(frame, label + " " + str(round(confidences[i], 2)), (x - 1, y - 10),
                #            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)

                # Nome do objeto, texto de cima
                #cv2.putText(frame, label + " " + str(round(confidences[i], 2)), (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1,
                #            cor, 1, lineType=cv2.LINE_AA)

            # Converte para jpg
            #cv2.rectangle(frame, (0,0), (frame.shape[1],60), (0,0,0), -1)
            #cv2.putText(frame, "Leitek-Demo1" + " " + "Time:" + " " + time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(time.time())) +  " " + "Azimuth:" +  " " + 					str(ptz.azimuth) + " " + "Elevation:"+ " " + str(ptz.elevation), (0, 45), 		 cv2.FONT_HERSHEY_DUPLEX, 1.25, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            #self.framecurrente = frame  # Esta var precisa de estar antes do codigo debaixo para ser funcional no firefox
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            print("Postprocessing5") 
            #frame = frame.tolist()
            # self.framecurrente = frame
            #frame=b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            # Guarda a imagem na Base Dados
            #if self.tempoPassado > 5:
            sector_azi=round(ptz.azimuth/15.0)+1
            if sector_azi>24:
               sector_azi=1
            if len(indexes) > 0:
               print(len(indexes))
               info=json.dumps({"id":"1", "sectorX":sector_azi, "sectorY":round(abs(ptz.elevation)/9.0)+1, "timestamp":time.time(),"azimuth":ptz.azimuth,"elevation":ptz.elevation, "objetos":objetos_capturados_frame}).encode('utf-8')
            #       self.tempoInicial = time.time()
            else:
               info=json.dumps({"id":"1", "sectorX":sector_azi, "sectorY":round(abs(ptz.elevation)/9.0)+1, "timestamp":time.time(),"azimuth":ptz.azimuth,"elevation":ptz.elevation}).encode('utf-8')
            #print("Fim processamento")
            length=len(frame)
            #print(length)
            global s1
            try:
               print("Postprocessing6")
               s1.sendall(length.to_bytes(8,'big'))
               s1.recv(1024).decode('ascii')
               print("Postprocessing7")
               s1.sendall(frame)
               s1.recv(1024).decode('ascii')
               s1.sendall(info)
               s1.recv(1024).decode('ascii')
               print("Postprocessing8")
            except socket.error:
               connected1 = False
               s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
               while not connected1:  
            # attempt to reconnect, otherwise sleep for 2 seconds  
                  try:
                     print("four")   
                     s1.connect((HOST2, PORT2))
                     s1.sendall(json.dumps({"id":1}).encode('utf-8'))
                     accept1=s1.recv(1024).decode('ascii')
                     print(accept1)
                     if accept1=='True':
                        connected1=True
                        print("I'm here")
                     else:
                        print("I'm here two")
                        s1.close()
                        s1=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        time.sleep( 2 )
                  except socket.error:  
                     time.sleep( 2 )
        else:
            print("Camara not connected")
            imagem.release()
            time.sleep(10)
            #return frame
               #videredb.guardaFrame(frame, self.id_user, time.time(), PTZ.azimuth, PTZ.elevation, confidences[0],objetos_captuados_frame)
            #    self.tempoPassado = 0
            #else:
            #    self.tempoPassado = time.time() - self.tempoInicial
Link="rtsp://root:2022Ltk@192.168.180.5:554/axis-media/media.amp?streamprofile=nvidia"
Nome="Axis"
class Server:
    def __init__(self):
       self.sock=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
       self.sock.bind((HOST, PORT))
       self.sock.listen()

    def listener(self, p, cmr):
       while True:
          conn, addr = self.sock.accept()     
          #print('Connected by', addr)
          while True:
             data = conn.recv(1024)
             if not data:
               break
             data = json.loads(data.decode('utf-8'))
             #print(data['linkCamara'])

                   #conn.sendall(json.dumps("ACK").encode('utf-8'))
                      #conn.sendall(json.dumps("ACK").encode('utf-8'))
                    #utilizador2.UTILIZADORES_ATIVOS[main.session["user_id"]].CriaCamara(linkCamara, nomeCamara) #filtroObjetos)
                      #s1.close()
             if data['tipo']=="move":
                  move(p,data['valor'], data['direcao'])
             elif data['tipo']=="mode":
                  cmr.mode=data['valor']
                  if data['valor']=="auto":
                     cmr.tour=data["tour"]
             elif data['tipo']=="sector":
                  cmr.mode="manual"
                  go(p,(int(data["X"])-1)*15,-9*(int(data["Y"])-1))
             elif data['tipo']=="go":
                  go(p,data['valor1'], data['valor2'])
             elif data['tipo']=="zoom":
                  zoom_in_out(p, data['valor'])
             elif data['tipo']=="speeddry":
                  start_speeddry(p)
             elif data['tipo']=="ircutfilter":
                  set_ircutfilter(p, data['valor'])
          
            #move(data['valor'], data['direcao'])

             '''with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST3, PORT3))
                s.sendall(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    #utilizador2.UTILIZADORES_ATIVOS[main.session["user_id"]].CriaCamara(linkCamara, nomeCamara) #filtroObjetos)
                s.close()'''
    def run(self,p):
          cThread = threading.Thread(target=self.listener, args=(p,cmr))
          #cThread.daemon = True
          cThread.start()
 
if __name__=="__main__":
    ai=AIEngine() 
    cmr=ai.CriaCamara(Link, Nome)
    p=ai.CriaPTZ(Link, Nome)
    cmr.stream(p) 
    #t1 = threading.Thread(target=cmr.stream, args=(p,))
    #t2 = threading.Thread(target=cmr.obtemFrame, args=(p,))
    #t3 = threading.Thread(target=p.getAzimuth)
    #t1.daemon = True
    #t2.daemon = True
    #t3.daemon = True
    #t1.start()
    #t2.start()
    #t3.start()
    #new_cam=json.dumps({"user":1,"link":cmr.id}).encode('utf-8')
    #length=len(new_cam)
    #s1.sendall(length.to_bytes(8,'big'))
    #ack1=s1.recv(1024).decode('ascii')
    #if ack1=="ACK":
    #   s1.sendall(new_cam)
    #server=Server()
    #server.run(p)
