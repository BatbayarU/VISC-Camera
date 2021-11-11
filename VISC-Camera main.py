#!/usr/bin/python3
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from time import gmtime, strftime, localtime
from datetime import datetime
import multiprocessing
from multiprocessing import Process, Value, Array, Queue, Pipe, Manager
from pylepton import Lepton
import pandas as pd
import dlib
import matplotlib.pyplot as plt
import pywt
from modwt import modwt, modwtmra
import heartpy as hp
import serial
import pywt as wt
from sklearn.decomposition import FastICA
from statsmodels.graphics.tsaplots import plot_acf
plt.style.use('ggplot')
font = cv2.FONT_HERSHEY_PLAIN
file_location = '/home/pi/shared/20210303/Data/'
face_cascade = cv2.CascadeClassifier('/home/pi/shared/20210303/haarcascade_frontalface_alt2.xml')
predictor = dlib.shape_predictor('/home/pi/shared/20210303/shape_predictor_68_face_landmarks.dat')
index = pd.read_csv("/home/pi/shared/20210303/index.csv", delimiter=",")
file_index = index['i'][0]# + 1
DATE = strftime("%Y-%m-%d_%H-%M", localtime())
filename_Realsense = str(file_index) + '-' + '.bag'
#filename_Depth = str(file_index) + '-' + DATE + '-Depth.avi'
filename_Thermal = str(file_index) + '-' + '-Thermal.avi'
#filename_RS_Time = str(file_index) + '-' + DATE + '-RS_Time.csv'
filename_Lep_Time = str(file_index) + '-' + '-Lepton_Time.csv'
RGB_fps = 30
Depth_fps = 30
Duration = 10
Recording_Control = False
Mouse_timer = 0
mouse_x = -1 #start - 670 back - 350
mouse_y = -1 #start - 320 back -320 child - 180-270
user_state  = 0
def get_click(event,x,y,flags,param):
    global mouse_x
    global mouse_y
    if event == cv2.EVENT_LBUTTONUP:
        mouse_x = x
        mouse_y = y
def show_result(measurement_state):
    Signals,axs = plt.subplots(4, 1,figsize=(3.4,4))
    Signals.subplots_adjust(top=0.92, bottom=0.13, left=0.10, right=0.95, hspace=0.6,wspace=0.35)
    Signals.canvas.set_window_title('Result Signals')
    Signals.canvas.manager.window.move(450,20)
    Signals.show()

    try:
        Heart = pd.read_csv("/home/pi/shared/20210303/Heartrate.csv", delimiter=",")
        heart1 = Heart['Wavelet-3']
        yf=np.fft.fft(Heart['Wavelet-3']/len(Heart['Wavelet-3']))
        yf=yf[range(int(len(Heart['Wavelet-3'])/2))]
        tc=len(Heart['Wavelet-3'])
        val=np.arange(int(tc/2))
        tp=tc/30
        fr=val/tp
        result_hr = str(round(60*fr[np.argmax(abs(yf))],1)) + ' beat per min'
    except Exception as ex:
        result_hr = 'NA'  
        heart1 = []
    try:
        Ref = pd.read_csv("/home/pi/shared/20210303/Reference.csv", delimiter=",")
        ref1 = Ref['H']
        ref2 = Ref['R']
    except Exception as ex:
        ref1 = []
        ref2 = []
        
    try:
        Resp = pd.read_csv("/home/pi/shared/20210303/Respiratory.csv", delimiter=",")
        resp1 = Resp['Bandpass']
        yf1=np.fft.fft(Resp['Bandpass']/len(Resp['Bandpass']))
        yf1=yf1[range(int(len(Resp['Bandpass'])/2))]
        tc1=len(Resp['Bandpass'])
        val1=np.arange(int(tc1/2))
        tp1=tc1/30
        fr1=val1/tp1
        result_resp = str(round(60*fr1[np.argmax(abs(yf1))],1)) + ' breath per min'
    except Exception as ex: 
        resp1 = [] 
        result_resp = 'NA'

    try:
        Temp = pd.read_csv("/home/pi/shared/20210303/Temperature.csv", delimiter=",")
        tempe1 = str(round(Temp['Temp'].mean(),1)) + ' C'
    except Exception as ex:  
        tempe1 = 'NA'
  
    axs[0].plot(heart1)
    axs[1].plot(ref1)
    axs[2].plot(resp1)
    axs[3].plot(ref2)

    img = np.zeros((480,480,3), np.uint8)
    cv2.putText(img, 'Temperature = ' + tempe1, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
    cv2.putText(img, 'Heartrate = ' + result_hr, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
    cv2.putText(img, 'Respiratory = ' + result_resp, (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
    cv2.namedWindow ("Result")
    cv2.moveWindow("Result", 0, 20)
    cv2.imshow("Result",img)
    cv2.waitKey(1)

    #plt.pause(0.1)
    plt.show()
    cv2.destroyWindow("Result")
    print("Result Stopped")

def convert_dlib_box_to_openCV_box(box):
    return (int(box.left()), int(box.top()), int(box.right() - box.left()),
                         int(box.bottom() - box.top()) )


def lepton(start_time, state,Face_ROI,body_temperature,Face_Found, option,measurement_state):
    Thermal_out = cv2.VideoWriter(filename_Thermal, cv2.VideoWriter.fourcc('X','V','I','D'), 30.0, (80,60))
    #Thermal_out = cv2.VideoWriter(filename_Thermal, cv2.VideoWriter.fourcc(*'RGBA'), 30.0, (80,60))
    start_time.value = time.time()
    b = np.zeros((60,80,1), 'uint8')
    c = np.zeros((60,80,1), 'uint8')
    body_temp = []
    thermal_raw = []
    cv2.namedWindow ("Thermal")
    cv2.moveWindow("Thermal", 430, 20)
    with Lepton() as l:
        while state.value == 1:
        #time.sleep(0.5)
            a,_ = l.capture()
            b = (a >> 8) & 0x00FF
            c = a & 0x00FF
            b = b.astype("uint8")
            c = c.astype("uint8")
            m = cv2.merge((b,b,c))
            thermal_roi = a[0:20,35:50]
            max_val = thermal_roi.max()
            body_temperature.value = (max_val/100 - 273.15)
            cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX) # extend contrast
            image = np.right_shift(a, 8, a) # fit data into 8 bits
            image = cv2.applyColorMap(np.uint8(a),cv2.COLORMAP_JET)
            #resized = cv2.resize(image,(400,300),interpolation = cv2.INTER_AREA)
            if Face_Found.value == 2 or option.value == 2:
                body_temp.append(body_temperature.value)
                thermal_raw.append(max_val)
            else:
                body_temp = []
                thermal_raw = []
            #cv2.rectangle(image, (35 , 0), (50, 20), (255,255,255), 1)
            cv2.imshow("Thermal",image)
            Thermal_out.write(image)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        Thermal_out.release()
        while measurement_state.value == 0:
            time.sleep(1)
        if measurement_state.value == 1 or option.value == 2:
            srR = pd.Series(thermal_raw, name='Raw')
            srT = pd.Series(body_temp, name='Temp')
            df = pd.concat([srR,srT], axis=1)
            df.to_csv(file_location + str(file_index) +'-' + '-Temperature.csv')   
            df.to_csv('/home/pi/shared/20210303/Temperature.csv')
    print("Thermal Display Stopped -> ID of process: {}".format(os.getpid()))
def realsense(start_time,frameBuffer, frameBuffer3D, Face_ROI, Face_Found, state, Distance, stable_track_count, Head_Pose, body_temperature,option,measurement_state):#, child_conn):
    print("Realsense is starting -> ID of process: {}".format(os.getpid()))
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
    #if option.value == 2:
    config.enable_stream(rs.stream.infrared, 424, 240, rs.format.y8, 30)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    config.enable_record_to_file(filename_Realsense)
    #
    queue = rs.frame_queue(600, keep_frames=True)
    #time.sleep(1)
    realsense = pipeline.start(config,queue)
    depth_sensor = realsense.get_device().query_sensors()[0]
    color_sensor = realsense.get_device().query_sensors()[1]
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    depth_scale = realsense.get_device().first_depth_sensor().get_depth_scale()
    start = time.time()
    
    timee = []
    frame_num = 0
    pre = 0
    
    pre_ff = 0

    cv2.namedWindow (str(file_index))
    cv2.moveWindow(str(file_index), 0, 20)
    m_time = 0
    p_time = 0
    start = time.time()
    process_start_time = time.time()
    break_status = 0
    while m_time < Duration and p_time < 50:
        frames = queue.wait_for_frame()#(timeout_ms=5)
        if frames.is_frameset():
            depth_frame = frames.as_frameset().get_depth_frame()
            color_frame = frames.as_frameset().get_color_frame()
            #ir_frame = frames.as_frameset().get_infrared_frame()
            #motion_frame = frames[2].as_motion_frame().get_motion_data()

            frame_number = frames.get_frame_number()
            if pre == frame_number:
                continue
            pre = frame_number
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            #ir_image = np.asanyarray(IR_frame.get_data())
            rgb_image = color_image.copy()
            if option.value == 3 or option.value == 2:    
                green_channel = rgb_image[:,:,1]
                shared_mem_size_color = frameBuffer.qsize()
                if shared_mem_size_color > 100:
                    print('---Color image process not responding!')
                    break_status = 1
                    break
                shared_mem_size_depth = frameBuffer.qsize()
                if shared_mem_size_depth > 100:
                    print('---Depth image process not responding!')
                    break_status = 2
                    break    
                green_crop = green_channel[0:240,130:294]
                depth_crop = depth_image[0:240,130:294]
                #ir_crop = IR_image[0:240,130:294]
            
                frameBuffer.put(green_crop.copy())
                frameBuffer3D.put(depth_crop)
            
            frame_num = frame_num + 1

            #if Face_Found.value == 1:
            a = Face_ROI[0] + 130
            b = Face_ROI[1]
            c = Face_ROI[2] + 130
            d = Face_ROI[3]
            Roll = Head_Pose[0]
            Yaw =Head_Pose[1]
            Pitch = Head_Pose[2]
            #disp_image = color_image.copy()
            
            if Face_Found.value == 2:
                cv2.rectangle(rgb_image, (a , b), (c, d), (0,255,0), 2)
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                cv2.circle(rgb_image, (int(abs((a + (c-a)/2))), int(b + (d-b)/2)), 1, (0, 255, 0), -1)
                cv2.putText(rgb_image, "Time = " + str(round(Duration - (time.time() - start))), (a + 5,b + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
                cv2.putText(rgb_image, "Temperature = " + str(round(body_temperature.value,1)), (a + 5,b + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)

                if Distance.value < 0.43 and Distance.value > 0.37:
                    cv2.putText(rgb_image, "Distance = " + str(round(Distance.value * 100,1)), (c + 5,b + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0),1)
                elif Distance.value > 0.45 or Distance.value < 0.35:
                    cv2.putText(rgb_image, "Distance = " + str(round(Distance.value * 100,1)), (c + 5,b + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1)
                else:
                    cv2.putText(rgb_image, "Distance = " + str(round(Distance.value * 100,1)), (c + 5,b + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
                cv2.putText(rgb_image, "Roll = " + str(Roll), (c + 5,b + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
                cv2.putText(rgb_image, "Yaw = " + str(Yaw), (c + 5,b + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
                cv2.putText(rgb_image, "Pitch = " + str(Pitch), (c + 5,b + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
            elif Face_Found.value == 1:
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                cv2.rectangle(rgb_image, (a , b), (c, d), (0, 0 + (stable_track_count.value*4),255 - (stable_track_count.value * 4)), 2)
                cv2.circle(rgb_image, (int(abs((a + (c-a)/2))), int(b + (d-b)/2)), 3, (0, 0 + (stable_track_count.value *4),255 - (stable_track_count.value * 4)), -1)
                cv2.putText(rgb_image, "Roll = " + str(Roll), (c + 5,b + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
                cv2.putText(rgb_image, "Yaw = " + str(Yaw), (c + 5,b + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
                cv2.putText(rgb_image, "Pitch = " + str(Pitch), (c + 5,b + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
                
                if pre_ff != 1:
                    pre_ff = Face_Found.value
                start = time.time()
                if Distance.value < 0.43 and Distance.value > 0.37:
                    cv2.putText(rgb_image, "Distance = " + str(round(Distance.value * 100,1)), (c + 5,b + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0),1)
                elif Distance.value > 0.45 or Distance.value < 0.35:
                    cv2.putText(rgb_image, "Distance = " + str(round(Distance.value * 100,1)), (c + 5,b + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1)
                else:
                    cv2.putText(rgb_image, "Distance = " + str(round(Distance.value * 100,1)), (c + 5,b + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
            else:
                start = time.time()
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                if pre_ff != 0:
                    pre_ff = Face_Found.value
                if Distance.value < 0.43 and Distance.value > 0.37:
                    cv2.putText(rgb_image, "Subject's Distance = " + str(round(Distance.value * 100,1)), (160,50), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0),1)
                elif Distance.value > 0.45 or Distance.value < 0.35:
                    cv2.putText(rgb_image, "Subject's Distance = " + str(round(Distance.value * 100,1)), (160,50), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1)
                else:
                    cv2.putText(rgb_image, "Subject's Distance = " + str(round(Distance.value * 100,1)), (160,50), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)
            cv2.circle(rgb_image, (82+130, 60), 10, (0, 0, 255), 1)
            cv2.imshow(str(file_index), rgb_image)
            k = 0
            k = cv2.waitKey(1)
            m_time = time.time() - start
            p_time = time.time() - process_start_time
            if k == 27:
                print('----- Measuremet stopped by user------')
                break_status = 3
                break
            pre_ff = Face_Found.value

    
    print(frame_num)
    print("Realsense Stopping -> ")
    state.value = 0
    pipeline.stop()
    while True:
        try:
            print("Realsense dequeue -> ")
            frames = queue.wait_for_frame(timeout_ms=100)
            frame_num = frame_num + 1
            print('After--------' + str(frame_num) + '--'+ str(frame_number))
        except:
            break
    print("Realsense Stopped -> ID of process: {}".format(os.getpid()))
    if m_time >= Duration:
        cv2.putText(rgb_image, "Measurement has done successfully", (50,70),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2)
        measurement_state.value = 1
    elif p_time >= 50:
        cv2.putText(rgb_image, "Measurement time has expired !!!", (50,70),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2)
        measurement_state.value = 2
    else:
        if break_status == 3:
            cv2.putText(rgb_image, "Measurement is stopped by user !!!", (50,70),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2)
        elif break_status == 2:
            cv2.putText(rgb_image, "Measurement is stopped due to Depth process !!!", (50,70),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2)
        elif break_status == 1:
            cv2.putText(rgb_image, "Measurement is stopped due to RGB process !!!", (50,70),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2)
        else:
            cv2.putText(rgb_image, "Measurement is stopped due to unknown error !", (50,70),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2)
        measurement_state.value = 3
    cv2.imshow(str(file_index), rgb_image)
    cv2.waitKey(1)
    time.sleep(5)
    cv2.destroyAllWindows()

def cvveiw(frameBuffer, Face_ROI, Face_Found,state,Distance,stable_track_count,Head_Pose, option, measurement_state):#, parent_conn):
    #pipeline.start(config)
    print("Display Started -> ID of process: {}".format(os.getpid()))
    start_time = time.time()

    frame_num = 0
    x,y,w,h = [0,0,0,0]
    tracking = False

    landmarks_initiated = False
    ysR = []
    ysL = []
    ysHarm = []

    stable_track_count.value = 0
    while state.value == 1 or not frameBuffer.empty() :

        if frameBuffer.empty():
            continue
        green_image = frameBuffer.get()
        if frame_num % 30 == 0 and not tracking and (Distance.value < 0.43 and Distance.value > 0.37):

            faces = face_cascade.detectMultiScale(green_image)
            if len(faces) != 0: # wait until first face is detected
                print('Face found')
                if True:#scores[0] > 1.0:
                    #bbox = convert_dlib_box_to_openCV_box(faces[0])
                    x, y, w, h = faces[0]
                    bbox = (x, y, w, h)
                    #cv2.circle(green_image, (int(abs((x + w/2))), int(y + h/2)), 3, (0, 0, 255), -1)
                    
                    if (abs((x + w/2) - 82) < 10 and abs((y + h/2) - 60) < 10):
                        Face_ROI[0] = int(x)
                        Face_ROI[1] = int(y)
                        Face_ROI[2] = int(x + w)
                        Face_ROI[3] = int(y + h)
                        
                        Face_Found.value = 1
                        tracker = cv2.TrackerMedianFlow_create()
                        ret = tracker.init(green_image, bbox)
                        if ret:
                            tracking = True
                            print('Tracker init OK')
                            #Face_Found.value = 2
                            mean_x = [x]*5
                            mean_y = [y]*5
                            roi_x = []
                            roi_y = []
                            roi_w = []
                            roi_h = []
                else:
                    Face_Found.value = 0
            else:
                Face_Found.value = 0
        elif tracking and (Distance.value < 0.45 and Distance.value > 0.35):
            ret, bbox = tracker.update(green_image)
            if ret:
                x, y, w, h = bbox
                
                del mean_x[0]
                del mean_y[0]
                mean_x.append(x)
                mean_y.append(y)
                x=(np.mean(mean_x))
                y=(np.mean(mean_y))
                
                Face_ROI[0] = int(x)
                Face_ROI[1] = int(y)
                Face_ROI[2] = int(x + w)
                Face_ROI[3] = int(y + h)
                
                if not landmarks_initiated and (Distance.value < 0.43 and Distance.value > 0.37) and (abs((x + w/2) - 82) < 10 and abs((y + h/2) - 60) < 10):
                    face =  dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    landmarks = predictor(green_image, face)
                    Roll = abs(landmarks.part(36).y - landmarks.part(45).y)
                    Yaw = abs(landmarks.part(27).x - landmarks.part(30).x)
                    Pitch = abs(landmarks.part(31).y - landmarks.part(30).y)
                    Head_Pose[0] = Roll
                    Head_Pose[1] = Yaw
                    Head_Pose[2] = Pitch
                    #for j in range(27,48):
                    #        cv2.circle(green_image, (int(landmarks.part(j).x), int(landmarks.part(j).y)), 3, (0, 0, 255), -1)
                    if (Roll < 3 and Yaw < 3 and Pitch < 20) or (option.value == 2):
                    #left_box_width = (mean_point_x[18] - mean_point_x[9])*1
                    #left_box_height = (mean_point_y[6] - mean_point_y[1])*1
                        if option.value == 3:
                            dist_x1 = landmarks.part(36).x - x - 10#landmarks.part(3).x - x#
                            dist_y1 = landmarks.part(40).y - y + 5
                            dist_x2 = landmarks.part(45).x - x +10 #landmarks.part(15).x - x#
                            dist_y2 = landmarks.part(30).y - y
                        else:
                            dist_x1 = landmarks.part(36).x - x - 10#landmarks.part(3).x - x#
                            dist_y1 = landmarks.part(40).y - y + 5
                            dist_x2 = landmarks.part(45).x - x +10 #landmarks.part(15).x - x#
                            dist_y2 = landmarks.part(33).y - y
                        
                        dis_x1 = x 
                        dis_y1 = y
                        #dis_x2 = mean_point_x[18]  #landmarks.part(15).x - x#
                        #dis_y2 = mean_point_y[3] 
                        '''
                        L_cheek = green_image[int(dist_y1+y):int(dist_y2+y), int(dist_x1+x):int(dist_x2+x)]
                        channels = cv2.mean(L_cheek)
                        hr = [channels[0]] * 100
                        
                        nose_shift = 0
                        bottom_shift = 1
                        yaw = 0
                        '''
                        landmarks_initiated = True
                        ysR = []
                        ysL = []
                        ysHarm = []
                        
                

                        '''
                        for j in range(27,48):
                            cv2.circle(green_image, (int(landmarks.part(j).x), int(landmarks.part(j).y)), 3, (0, 0, 255), -1)
                    
                        '''
                elif landmarks_initiated and (Distance.value < 0.45 and Distance.value > 0.35) and (abs((x + w/2) - 82) < 10 and abs((y + h/2) - 60) < 10):
                    if stable_track_count.value > 60:
                        L_cheek = green_image[round(dist_y1+y+1):round(dist_y2+y-1), round(dist_x1+x+1):round(dist_x2+x-1)] # + (h*20/100)

                        channels = cv2.mean(L_cheek)
                        ysL.append(channels[0])

                        roi_x.append(round(x))
                        roi_y.append(round(y))
                        roi_w.append(round(w))
                        roi_h.append(round(h))
                        
                        Face_Found.value = 2
                        #cv2.rectangle(green_image,(int(dist_x1+x),int(dist_y1+y)),(int(dist_x2+x),int(dist_y2+y)), (255, 255, 255), 1)
                        #cv2.rectangle(green_image,(int(dist_x1+x+1),int(dist_y1+y+1)),(int(dist_x2+x-1),int(dist_y2+y-1)), (255, 255, 255), 1)
                    else:
                        stable_track_count.value = stable_track_count.value + 1
                        Face_Found.value = 1
                else:
                    stable_track_count.value = 0
                    Face_Found.value = 1
                #cv2.circle(green_image, (int(abs((x + w/2))), int(y + h/2)), 1, (0, 0, 255), -1)
            else:
                tracking = False
                Face_Found.value = 1
                Face_ROI[0], Face_ROI[1], Face_ROI[2], Face_ROI[3] = [0,0,0,0]
                landmarks_initiated = False
                stable_track_count.value = 0
                print('-Tracker Failed')
        else:
            if tracking:
                print('-Out of Range')
            tracking = False
            Face_Found.value = 0
            Face_ROI[0], Face_ROI[1], Face_ROI[2], Face_ROI[3] = [0,0,0,0]
            landmarks_initiated = False
            stable_track_count.value = 0
            
        frame_num = frame_num + 1
        '''
        cv2.imshow(' Color', green_image)
        cv2.waitKey(1)
        '''
    
    while measurement_state.value == 0:
        time.sleep(1)
    #print(filtered)
    #filtered1 = hp.filter_signal(ysR, [0.7, 2.5], sample_rate=30, order=3, filtertype='bandpass')
    #temp = pd.Series(filtered1)
    #temp = temp.diff(10)
    #temp = temp.drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #ysR = temp.values.tolist()
    #filtered = np.asanyarray(filtered)
    #print(filtered)
    if measurement_state.value == 1:
        filtered = hp.filter_signal(ysL, [0.7, 2.5], sample_rate=30, order=3, filtertype='bandpass')
        wt = modwt(filtered, 'sym2', 5)
        wtmra = modwtmra(wt, 'sym2')
        
        srX = pd.Series(ysL, name='Raw')
        srY = pd.Series(filtered, name='Bandpass')
        srW = pd.Series(wtmra[3], name='Wavelet-3')
        srR_x = pd.Series(roi_x, name='roi_x')
        srR_y = pd.Series(roi_y, name='roi_y')
        srR_w = pd.Series(roi_w, name='roi_w')
        srR_h = pd.Series(roi_h, name='roi_h')
        df = pd.concat([srX,srY,srW, srR_x,srR_y,srR_w,srR_h], axis=1)
        df.to_csv(file_location + str(file_index) +'-' + '-Heartrate.csv')
        df.to_csv('/home/pi/shared/20210303/Heartrate.csv')
    #print(frame_num)
    #cv2.destroyAllWindows()
    
    print("Gray Display Stopped -> ID of process: {}".format(os.getpid()))
def cv3Dveiw(frameBuffer3D,Face_ROI, Face_Found,state,Distance, option, measurement_state):#, parent_conn):
    #pipeline.start(config)
    print("Display Started -> ID of process: {}".format(os.getpid()))
    start_time = time.time()
    frame_num = 0
    resp = []
    distance= []
    body_depth = []
    roi_set = False
    initial_s = 0
    while state.value == 1 or not frameBuffer3D.empty():

        if frameBuffer3D.empty():
            continue

        depth_image = frameBuffer3D.get_nowait()
        frame_num = frame_num + 1
        clipping_distance = 1/0.0010000000474974513
        #depth_image = cv2.medianBlur(depth_image,5)
        bg_removed1 = np.where((depth_image > clipping_distance) | (depth_image <= 0), clipping_distance, depth_image)
        #bg_removed1 = np.where((depth_image > clipping_distance), clipping_distance, depth_image)
        min_val = bg_removed1.min()
        bg_removed2 = np.where((bg_removed1 >= clipping_distance), 0, bg_removed1)
        max_val = bg_removed2.max()
        bg_removed = np.where((depth_image > max_val)| (depth_image <= 0), max_val, depth_image)
        #bg_removed = np.where((bg_removed == 0), min_val, bg_removed)
        #print('Max and min vals -----------' + str(max_val) + '---'+str(min_val))
        #diff = max_val - min_val
        #print("---body depth = " + str(diff))
        #bg_removed = np.where((bg_removed < min_val), min_val, bg_removed)
        #bg_removed = max_val - bg_removed
        #bg_removed = cv2.medianBlur(bg_removed,10)
        
        #bg_removed = cv2.medianBlur(bg_removed,3)

        a = Face_ROI[0] 
        b = Face_ROI[1]
        c = int(Face_ROI[0] +((Face_ROI[2] - Face_ROI[0])/5)*2)
        d = int(Face_ROI[1] +((Face_ROI[3])))# - Face_ROI[1])/3)*1)
        #cv2.rectangle(bg_removed, (62 , 40), (82, 60), (255,255,255), 1)
        object_roi = bg_removed[40:60, 62:82]
        obj_roi_min = object_roi.min() * 0.0010000000474974513
        obj_roi_max = object_roi.max() * 0.0010000000474974513
        print(obj_roi_min)
        if (obj_roi_min < 0.46 and obj_roi_min > 0.36) and  obj_roi_max == max_val:
            object_roi = np.where((object_roi == obj_roi_max), obj_roi_min, object_roi)
        #print(obj_roi_max* 0.0010000000474974513)
        obj_roi_max = object_roi.max()
        Distance.value = obj_roi_min #* 0.0010000000474974513
        
        #print('\t',"Distanse = " + str(mean_val[0]) + 'm', end='',flush=True)
        if Face_Found.value == 2:
            #cv2.rectangle(bg_removed, (a , b), (c, d), (255,255,255), 1)#164x240
            #cv2.rectangle(bg_removed, (31 , 160), (131, 240), (255,255,255), 1)
            #cv2.rectangle(bg_removed, (72 , d +5), (92, d +15), (255,255,255), 1)
            #face_roi11 = bg_removed[d + 10:240, a:c]
            face_roi = bg_removed[160:240, 31:131]
            ref_roi = bg_removed[d +5:d + 15, 72:92]
            ref_roi = np.where((ref_roi == max_val), min_val, ref_roi)
            ref_roi_mean = ref_roi.max()
            #ref_roi = np.where((ref_roi == max_val), ref_roi_mean[0], ref_roi)
            #ref_roi_mean = cv2.mean(ref_roi)
            #roi_max_val = ref_roi.max()
            roi_min_val = face_roi.min()
            #diff = roi_max_val - roi_min_val
            #print("---body depth = " + str(diff))
            #roi_mean_val = cv2.mean(face_roi)
            face_roi = np.where((face_roi == max_val), roi_min_val, face_roi)
            #ref_roi = np.where((ref_roi == max_val), roi_max_val, ref_roi)
            face_roi = face_roi * 0.0010000000474974513
            s = ((2*obj_roi_max * np.tan(43))/424) * ((2* obj_roi_max  * np.tan(57/2))/240)
            s = abs(s)
            if roi_set == False:
                initial_s = s
                roi_set = True
            delta_s = s/initial_s

            #mean_val = cv2.mean(bg_removed)
            #mean_Length = ((mean_val * np.tan(np.deg2rad(43))/212))
            #mean_Heigth = ((mean_val * np.tan(np.deg2rad(57/2))/120))
            face_roi = ((obj_roi_max)  * 0.0010000000474974513 ) - face_roi #+ max_val - min_val
            
            face_roi = face_roi * s 
            
            #print('\t',"Distanse = " + str(mean_val[0]) + 'm', end='',flush=True)

            resp.append((np.sum(face_roi)))
            distance.append(obj_roi_max * 0.0010000000474974513)# mean_val[0])
            body_depth.append(roi_min_val * 0.0010000000474974513)
            #face_roi = cv2.applyColorMap(cv2.convertScaleAbs(face_roi, alpha=0.001), cv2.COLORMAP_JET)
            #cv2.imshow(' roi', face_roi)
            #cv2.waitKey(1)
        else:
            resp = []
            distance= []
            body_depth = []
            roi_set = False

        #resized_color = cv2.resize(color_image,(400,300),interpolation = cv2.INTER_AREA)
        #bg_removed = cv2.applyColorMap(cv2.convertScaleAbs(bg_removed, alpha=0.3), cv2.COLORMAP_JET)
        #cv2.imshow(' 3D', bg_removed)
        #cv2.waitKey(1)
    while measurement_state.value == 0:
        time.sleep(1)
    if measurement_state.value == 1:
        filtered = hp.filter_signal(resp, [0.1, 1], sample_rate=30, order=3, filtertype='bandpass')
        f_dis = hp.filter_signal(distance, [0.01, 1], sample_rate=30, order=3, filtertype='bandpass')
        d_dis = hp.filter_signal(body_depth, [0.01, 1], sample_rate=30, order=3, filtertype='bandpass')
        srY = pd.Series(resp, name='Raw')
        srDr= pd.Series(distance, name = 'Dist Raw')
        srD = pd.Series(f_dis, name='Dist')
        srE = pd.Series(filtered, name='Bandpass')
        srM = pd.Series(d_dis, name='Min')
        df = pd.concat([srY, srDr, srD, srE, srM], axis=1)
        df.to_csv(file_location + str(file_index) +'-' + '-Respiratory.csv')   
        df.to_csv('/home/pi/shared/20210303/Respiratory.csv')  

    #cv2.destroyAllWindows()
    
    print("3D Display Stopped -> ID of process: {}".format(os.getpid()))

def DAQ(start_time,state,Heart,Respiratory,Face_Found, option,measurement_state):
    try:
        print("DAQ Started -> ID of process: {}".format(os.getpid()))  
        hr = [0] * 100
        rr = [0] * 100
        HearRate_PPG = []
        Respiratory_Belt = []
        pre_Face_Found = Face_Found.value
        plt.ion()
        Signals,axs = plt.subplots(2, 1,figsize=(3.4,2))
        Signals.subplots_adjust(top=0.92, bottom=0.13, left=0.10, right=0.95, hspace=0.6, wspace=0.35)
        axs[0].set_title('PPG Signal')
        axs[1].set_title('Respiratory Belt Signal')
        line_0, = axs[0].plot(hr,'-', alpha=0.5)
        line_1, = axs[1].plot(rr,'-', alpha=0.5)
        axs[0].grid(True)
        axs[1].grid(True)
        Signals.canvas.set_window_title('Reference Signals')
        Signals.canvas.manager.window.move(450,220)
        Signals.show()

        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        ser.flush()
        while state.value == 1:
            while ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='replace').rstrip()
                now = time.time()
                x = line.find(".")
                y = line.find(",")
                if x != -1 and y != -1:       
                    del hr[0]
                    del rr[0]
                    try:
                        hr.append(int(line[x+1:y]))                    
                    except Exception as ex:
                        hr.append(int(0))
                    try:
                        rr.append(int(line[y+1:]))
                    except Exception as ex:
                        rr.append(int(0))
                    if Face_Found.value == 2 or option.value == 2:
                        HearRate_PPG.append(hr[99])
                        Respiratory_Belt.append(rr[99])

                    elif pre_Face_Found == 2:
                        HearRate_PPG = []
                        Respiratory_Belt = []
            pre_Face_Found = Face_Found.value
            line_0.set_ydata(hr)
            line_1.set_ydata(rr)
            axs[0].set_ylim([np.min(hr)-5,np.max(hr)+20])
            axs[1].set_ylim([np.min(rr)-5,np.max(rr)+20])
            plt.pause(0.0001)
        while measurement_state.value == 0:
            time.sleep(1)
        if measurement_state.value == 1 or option.value == 2:
            if measurement_state.value == 1 and option.value == 2:
                HearRate_PPG = HearRate_PPG[len(HearRate_PPG)-330:len(HearRate_PPG)-1]
                Respiratory_Belt = Respiratory_Belt[len(Respiratory_Belt)-330:len(Respiratory_Belt)-1]
            srX = pd.Series(HearRate_PPG, name='H')
            srY = pd.Series(Respiratory_Belt, name='R')
            df = pd.concat([srX,srY], axis=1)
            df.to_csv(file_location + str(file_index) +'-' + '-Reference.csv')
            df.to_csv('/home/pi/shared/20210303/Reference.csv')
        print("DAQ Stopped -> ID of process: {}".format(os.getpid()))
    except Exception as ex:
        print("DAQ Stopped by Exception -> ID of process: {}".format(os.getpid()))
def wait(option):
    global mouse_x  #start - 670 back - 350
    global mouse_y  #start - 320 back -320 child - 180-270 
    global user_state
    img = np.zeros((512,512,3), np.uint8)
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('image',get_click)
    start = cv2.imread('/home/pi/shared/20210303/start.png',-1)
    select = cv2.imread('/home/pi/shared/20210303/select.png',-1)
    child = cv2.imread('/home/pi/shared/20210303/child.png',-1)
    adult = cv2.imread('/home/pi/shared/20210303/adult.png',-1)
    while True:
        if user_state == 0:
            cv2.imshow('image',start)
            if mouse_x > 700 and mouse_y > 360:
                user_state = 1
            #elif mouse_x < 100 and mouse_y > 360:
                #break
            mouse_x = -1
            mouse_y = -1
        elif user_state == 1:
            cv2.imshow('image',select)
            if mouse_x > 700 and mouse_y > 360:
                user_state = 3
            if mouse_x > 700 and (mouse_y < 260 and mouse_y > 220):
                user_state = 2
            elif mouse_x < 100 and mouse_y > 360:
                user_state = 0
            mouse_x = -1
            mouse_y = -1
        elif user_state == 2:
            cv2.imshow('image',child)
            if mouse_x > 700 and mouse_y > 360:
                break
            elif mouse_x < 100 and mouse_y > 360:
                user_state = 1
            mouse_x = -1
            mouse_y = -1
        else:
            cv2.imshow('image',adult)
            if mouse_x > 700 and mouse_y > 360:
                break
            elif mouse_x < 100 and mouse_y > 360:
                user_state = 1
            mouse_x = -1
            mouse_y = -1
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    option.value = user_state
    cv2.destroyWindow("image")

if __name__=="__main__":
    counter = 0
    while True: 
        option = Value('i', 0)
        Heart       = Array('f',1000000)
        Respiratory = Array('f',1000000)
        Time_DAQ    = Array('f',1000000)
        Time_CAM    = Array('f',1000000)
        start_time  = Value('d', 0)
        measurement_state   = Value('i', 0)
        Face_ROI    = Array('i', 4)
        Head_Pose   = Array('i', 4)
        Face_Found  = Value('i', 0)
        index       = Value('i', 0)
        index_CAM   = Value('i', 0)
        Distance    = Value('f', 0)
        state       = Value('i', 9)
        non         = Value('i', 9)
        #start_time  = Value('d', 0)
        body_temperature = Value('f',0)
        frameBuffer = Queue(600)
        frameBuffer3D = Queue(600)
        Face_ROI[0], Face_ROI[1], Face_ROI[2], Face_ROI[3] = [0,0,0,0]
        Head_Pose[0], Head_Pose[1], Head_Pose[2], Head_Pose[3] = [0,0,0,0]
        state.value = 1
        stable_track_count = Value('i',0)
        #parent_conn, child_conn = Pipe()
        print("ID of main process: {}".format(os.getpid()))
        if counter > 0:
            p7 = Process(target = show_result,args=(measurement_state,))
            p7.start()
            p7.join()

        p0 = Process(target = wait,args=(option,))
        p0.start()
        p0.join()
        print(option.value)
        REC         = Value('i', 0)
        SYSTEM_START= Value('i', 1)
        if option.value == 2 or option.value == 3:
            
            index = pd.read_csv("/home/pi/shared/20210303/index.csv", delimiter=",")
            file_index = index['i'][0]# + 1
            DATE = strftime("%Y-%m-%d_%H-%M", localtime())
            filename_Realsense = file_location + str(file_index) + '-' + '.bag'
            #filename_Realsense = os.path.join(filename_Realsense, str(file_index) + '-' + '.bag')
            #filename_Depth = str(file_index) + '-' + DATE + '-Depth.avi'
            filename_Thermal = file_location + str(file_index) + '-' + '-Thermal.avi'
            #filename_RS_Time = str(file_index) + '-' + DATE + '-RS_Time.csv'
            #filename_Lep_Time = '/home/pi/shared/20210303/Data/' + str(file_index) + '-' + '-Lepton_Time.csv'
            #value = input("Please enter a string\n")
            p1 = Process(target = realsense, args=(start_time, frameBuffer, frameBuffer3D, Face_ROI, Face_Found,state,Distance,stable_track_count,Head_Pose,body_temperature,option,measurement_state,))# child_conn,))
            p2 = Process(target = lepton, args=(start_time,state,Face_ROI,body_temperature,Face_Found, option,measurement_state,))
            #p6 = Process(target = show_result, args = (measurement_state,))

            p3 = Process(target = cvveiw, args=(frameBuffer, Face_ROI, Face_Found,state,Distance,stable_track_count,Head_Pose, option,measurement_state,))# parent_conn,))
            p4 = Process(target = cv3Dveiw, args=(frameBuffer3D,Face_ROI, Face_Found, state,Distance, option,measurement_state,))
            p5 = Process(target = DAQ, args=(start_time,state,Heart,Respiratory,Face_Found, option,measurement_state,))
            start_time.value = time.time()
            p1.start()
            p2.start()
            #p6.start()
            p3.start()
            p4.start()
            p5.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
            #p6.join()
            print("ALL DONE")
            if measurement_state.value == 1 or (measurement_state.value == 2 and option.value == 2):
                print('Measurement number- ' + str(file_index))
                file_index = file_index + 1
                sr = pd.Series(file_index, name='i')
                df = pd.concat([sr], axis=1) 
                df.to_csv("/home/pi/shared/20210303/index.csv")
                counter = counter + 1
            else:
                print('Measurement number- ' + str(file_index))    
            print("ALL DONE")

        else:
            break

    

















 
