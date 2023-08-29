import numpy as np
import cv2
import time
from src.tracking.byte_track import BYTETracker
from src.detector.Yolo_detect import Detector
from src.MMC.color_recognition import Color_Recognitiion
from src.OCR.text_recognizer import TextRecognizer
import time
import math
p1, p2,p3,p4 = None, None,None,None
state = 0
frame =None
crop = False
CLASS_NAME=["bus","car","person","trailer","truck"]
CLASS_NAME2=["bus","car","lane","person","trailer","truck","bike"]
def get_area_detect(img, points):
    # points = points.reshape((-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dts = cv2.bitwise_and(img, img, mask=mask)
    return dts
# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2,p3,p4
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Select first point
        if state == 0:
            p1 = [x,y]
            state += 1
        # Select second point
        elif state == 1:
            p2 = [x,y]
            state += 1
          # Select second point
        elif state == 2:
            p3 = [x,y]
            state += 1
          # Select second point
        elif state == 3:
            p4 = [x,y]
            state += 1
    # Right click (erase current ROI)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        p1, p2,p3,p4 = None, None,None,None
        state = 0

class Args():
    def __init__(self) -> None:
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.7
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = True
        self.tsize = None
        self.exp_file = None
class Tracking():
    def __init__(self) -> None:
        super(Tracking,self).__init__()
        args=Args()
        self.tracker = BYTETracker(args, frame_rate=22)
        self.vehicle_detector = Detector()
        self.plate_detector=Detector()
        self.hetmet_trafficlight=Detector()
        self.color_recog=Color_Recognitiion()
        self.lp_recog=TextRecognizer()
        self.test_size=(640,640)
    def __four_points_transform(self, image):
        H, W, _= image.shape
        rect = np.asarray([[0, 0], [W ,0], [W, H], [0, H]], dtype= "float32")
        width = 195
        height = 50
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype= "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(image, M, (width, height))

        return warp

    def __norm_plate(self, iplate, classID=None):
        if iplate is None:
            return []
        h, w, _ = iplate.shape
        if w / h > 2.5:
            iplate_transform = self.__four_points_transform(iplate)

            return iplate_transform
        else:
            iplate_1 = iplate[0:int(h / 2), 0:w]
            iplate_2 = iplate[int(h / 2):h, 0:w]
            _iplate_1 = cv2.resize(iplate_1, (165, 50))
            _iplate_2 = cv2.resize(iplate_2, (165, 50))
            iplate_concat = cv2.hconcat([_iplate_1, _iplate_2])
            iplate_blur = cv2.GaussianBlur(iplate_concat, (7, 7), 0)

            return iplate_blur

    def __rule(self, text):
        text_new = ''
        text = ''.join(char for char in text if char.isalnum() or char.isalpha())
        text = text.upper()
        arr = list(text)
        if len(arr) == 9 and arr[3].isalpha() and not arr[2].isalpha():
            arr = arr[1:] 
        if len(arr) > 6:
            if arr[2] in '8':
                arr = arr[:2] + ['B'] + arr[2+1:]
            if arr[0] in 'B':
                arr = ['8'] + arr[1:]
            if arr[1] in 'B':
                arr = arr[:1] + ['8'] + arr[1+1:]
            # 7 Z
            if arr[0] in 'Z':
                arr = ['7'] + arr[1:]
            if arr[1] in 'Z':
                arr = arr[:1] + ['7'] + arr[1+1:]
            # 0 D
            if arr[2] in '0':
                arr = arr[:2] + ['D'] + arr[2+1:]
            # A 4
            if arr[2] in '4':
                arr = arr[:2] + ['A'] + arr[2+1:]
            # 7 V
            if arr[0] in 'V':
                arr = ['7'] + arr[1:]
            if arr[1] in 'V':
                arr = arr[:1] + ['7'] + arr[1+1:]
            # O D
            if arr[3] in 'O':
                arr = arr[:3] + ['D'] + arr[3+1:]
            text_new = ''.join(str(elem) for elem in arr)
        
        return text_new
    
    def color_recognition(self,img,bbox_car):
        '''
        Nhận diện màu xe ,cắt các vùng ảnh chứa xe r cho vào nhận diện 
        '''
        color_list=[]
        for box in bbox_car :
            box=list(map(int,box))
            car_img_crop=img[box[1]:box[3],box[0]:box[2]]
            color=self.color_recog.infer(car_img_crop)
            color_list.append(color)
        return color_list
    def plate_recognition(self,img,bbox_vehicle):
        '''
        Phát hiện phương tiện phải chạy liên tục
        sau đó đến điểm cần detect biển thì chạy detect biển 
        '''
        pass
    def hemet_detector(self,img,bbox_person):
        '''
        Phát hiện người 
        sau đó crop vùng người trong ảnh r cho vào mô hình 
        '''
        for box in bbox_person:
            box=list(map(int,box))
            person_img=img[box[1]:box[3],box[0]:box[2]]
            output,bbox=self.hetmet_trafficlight.detect(person_img)
            cls=output[:,5]
        return cls
    
    def trafic_detection(self,img,cls_traffic_light):
        '''
        Phát hiện trên từng frame hình 
        đưa ra nhãn 
        '''
        pass

    def infer(self,img:np.ndarray):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        # create filter class
        ratio =1  
        outputs,bbox=self.detector.detect(img)
        # print("img shape :",img.shape)
        scores=outputs[:,4]
        cls=outputs[:,5]
        img_info["ratio"] = ratio
        filter_class = [0,1,4,5,6]
        list_count=[0,0,0,0]
        if outputs is not None:
            online_targets = self.tracker.update(outputs, [height,width], self.test_size, filter_class)
            online_tlwhs = []
            online_ids = []
            for t,cl in zip(online_targets,cls):
                tlwh = t.tlwh
                tid = t.track_id              
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        return bbox,cls,online_ids,scores

if __name__ == '__main__':
    cap=cv2.VideoCapture("./video/video5.avi")
    width=int(cap.get(3))
    height=int(cap.get(4))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img = np.zeros((1280,720,3), np.uint8)
    track=Tracking()
    count=0
    check=0
    MIN=40
    list_count_left=[0,0,0,0]
    list_count_right=[0,0,0,0]
    start_point={}
    check_point={}
    M_left={"bus":[],"car":[],"lane":[],"person":[],"trailer":[],"truck":[],"bike":[]}
    M_right={"bus":[],"car":[],"lane":[],"person":[],"trailer":[],"truck":[],"bike":[]}
    Max_contours=0
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    result = cv2.VideoWriter('file.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1280,720))
    while True:
        update_point={}
        _,frame=cap.read()
        c3_new=[]
        count+=1
        if count>length:
            break
        frame=cv2.resize(frame,(1280,720))
        img=frame.copy()
        start_time=time.time()
        # Cropping image
        if p1 is not None and p2 is not None and p3 is not None and p4 is not None :
            pts=np.array([p1,p2,p3,p4],np.int32)
            #crop frame 
            img_croped=get_area_detect(img,pts)
            cv2.polylines(img,[pts],True,(0,0,142),3)
            img_copy=img_croped.copy()
            #Tracking
            bbox,cls,id_list,scores=track.infer(img_croped)
            arr_point=[p1,p2,p3,p4]
            #sort array from y
            arr_point.sort(key=lambda x:x[1])
            arr_point2=[p1,p2,p3,p4]
            arr_point2.sort()
            p1_2,p2_2,p3_2,p4_2=arr_point2
            p1_1,p2_1,p3_1,p4_1=arr_point
            point_1=[int((arr_point[0][0]+arr_point[1][0])/2),int((arr_point[0][1]+arr_point[1][1])/2)]
            point_2=[int((arr_point[2][0]+arr_point[3][0])/2),int((arr_point[2][1]+arr_point[3][1])/2)]
            center=[point_1,point_2]
            cv2.line(img,point_1,point_2,(0,255,0),4)
            
            cv2.putText(img,"LANE 1",(max(point_1[0],point_2[0])-100,point_1[1]+100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(210,120,10),1)
            cv2.putText(img,"LANE 2",(max(point_1[0],point_2[0])+50,point_1[1]+100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(210,120,10),1)
            '''
            Viết phương trình đường thẳng  
            '''
            # Phương trình đường thẳng AB với A=p1_2 , B=p2_2
            x1,y1=p1_2
            x2,y2=p2_2
            a_AB=((y1-y2))/(x1-x2)
            b_AB=(y2*x1-y1*x2)/(x1-x2)
            y_M=max(y1,y2)*0.6
            x_M=(y_M-b_AB)/a_AB
            K=[int(x_M),int(y_M)]
            # Phương trình đường thẳng CD với C=p3_2 D=p4_2
            x3,y3=p3_2
            x4,y4=p4_2
            a_CD=((y3-y4))/(x3-x4)
            b_CD=(y4*x3-y3*x4)/(x3-x4)
            y_N=max(y3,y4)*0.6
            x_N=(y_N-b_CD)/a_CD
            N=[int(x_N),int(y_N)]
            area_Goal2=np.array([K,N,p3_1,p4_1],np.int32)
            area_Goal=np.array([[p4_1[0],p4_1[1]-150],[p3_1[0],p3_1[1]-150],p3_1,p4_1],dtype=np.int32)
            cv2.polylines(img,[area_Goal],True,(0,125,125),4)

            for box,cl,id,sc in zip(bbox,cls,id_list,scores):
                cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,125),2)

                    
                    # cv2.putText(img,str(id),(int(box[0]),int(box[1])-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                if int(cl)==3:
                    cv2.putText(img,"Person : Found ",(10,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                

                
                if  int(cl)!=2 :
                    cv2.putText(img,CLASS_NAME2[int(cl)],(int(box[2]),int(box[3])-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                    cv2.putText(img,str(id),(int(box[0]),int(box[1])-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                    mid_point=(int((box[0]+box[2])/2),int((box[1]+box[3])/2))
                    if str(id) not in start_point.keys():
                        start_point[str(id)]=[mid_point,time.perf_counter()]
                        # print("Add object to dict")
                    update_point[str(id)]=[mid_point,time.perf_counter()]
                    
                    '''
                    Phân chia làn 
                    '''
                    if mid_point[0]<max(point_1[0],point_2[0]):
                        if str(id) not in M_left[CLASS_NAME2[int(cl)]] :
                            M_left[CLASS_NAME2[int(cl)]].append(str(id))
                        print(" Đối tượng đang ở Làn 1 ")
                        list_count_left=[len(M_left["car"]),len(M_left["bus"]),len(M_left["trailer"]),len(M_left["truck"])]
                    else :
                        if str(id) not in M_right[CLASS_NAME2[int(cl)]] :
                            M_right[CLASS_NAME2[int(cl)]].append(str(id))
                        print("Đối tượng đang ở làn 2")
                        list_count_right=[len(M_right["car"]),len(M_right["bus"]),len(M_right["trailer"]),len(M_right["truck"])]
                        
                    mid_point_t=start_point[str(id)][0]
                    t=start_point[str(id)][1]
                    mid_point_t_1=update_point[str(id)][0]
                    
                    t_1=update_point[str(id)][1]-t
                    
                    distance_pixel=math.hypot(abs(mid_point_t[0]-mid_point_t_1[0]),abs(mid_point_t[1]-mid_point_t_1[1]))
                    if distance_pixel<MIN and t_1>5:
                        cv2.putText(img,"Stopped",(int(box[0]),int(box[1]+10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,12,244),1)
                    results_goal=cv2.pointPolygonTest(area_Goal,mid_point,False)
                    if results_goal>=0:
                        mid_point_prev=start_point[str(id)][0]
                        start_time=start_point[str(id)][1]
                        distance_pixel=math.hypot(abs(mid_point[0]-mid_point_prev[0]),abs(mid_point[1]-mid_point_prev[1]))
                        end_time=update_point[str(id)][1]-start_time
                        print("Time",end_time)
                        print("Đối tượng {} chuẩn bị thoát ra khỏi vùng kiểm soát".format(id))
                        distance_const=distance_pixel*0.3 # m
                        velocity=(distance_const/end_time)*3.6
                        print("velocity : ",velocity)
                        print("Khoảng cách đối tượng di chuyển trong vùng quan sát là :",distance_const)
                        cv2.putText(img,"{:.2f} km/h".format(velocity),(int(box[0]),int(box[1]+10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,12,14),1)         
        # result.write(img)
        #  
        cv2.imshow('frame',img)
        ##### Clear dict update_point_t1 sau khi đã sử dụng ###########
        
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break