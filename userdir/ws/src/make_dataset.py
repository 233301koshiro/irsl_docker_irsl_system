#jupyter console --kernel=choreonoid
#exec(open('make_dataset.py').read())
exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())

import random
random.seed()
import torch
import numpy as np
import cv2
dirname=''#userdirにロボットモデルあるしいいんじゃない？ｎ
import os
from scipy import stats
os.environ["ROS_PACKAGE_PATH"] = '/userdir/example-robot-data:' + os.environ["ROS_PACKAGE_PATH"]

#四足ロボット
#四足ロボット
list
lst_a1 = [
    ['example-robot-data/robots/a1_description/urdf/a1.urdf' ,
     [ ('fl', 'FL_foot', None), ## (name, link_name, tip_link_to_eef)
       ('fr', 'FR_foot', None),
       ('rl', 'RL_foot', None),
       ('rr', 'RR_foot', None),
      ]
     ],
    ###
    ]

lst_b1_z1 = [
    ['example-robot-data/robots/b1_description/urdf/b1-z1.urdf' ,
     [ ('lf', 'FL_foot', None), ## (name, link_name, tip_link_to_eef)
       ('fr', 'FR_foot', None),
       ('rl', 'RL_foot', None),
       ('rr', 'RR_foot', None),
       ('gm_up', 'gripperMover', None),
       ('gm_dn', 'gripperMover', None),
      ]
     ],
    ###
    ]

lst_b1 = [
    ['example-robot-data/robots/b1_description/urdf/b1.urdf' ,
     [ ('fl', 'FL_foot', None), ## (name, link_name, tip_link_to_eef)
       ('fr', 'FR_foot', None),
       ('rl', 'RL_foot', None),
       ('rr', 'RR_foot', None),
      ]
     ],
    ###
    ]
lst_go1 = [
    ['example-robot-data/robots/go1_description/urdf/go1.urdf' ,
     [ ('fl', 'FL_foot', None), ## (name, link_name, tip_link_to_eef)
       ('fr', 'FR_foot', None),
       ('rl', 'RL_foot', None),
       ('rr', 'RR_foot', None),
      ]
     ],
    ###
    ]

lst_romeo_small = [
    ['example-robot-data/robots/romeo_description/urdf/romeo_small.urdf' ,
     [ ('rg', 'r_gripper', None), ## (name, link_name, tip_link_to_eef)
       ('lg', 'l_gripper', None),
       ('rs', 'r_sole', None),
       ('ls', 'l_sole', None),
      ]
     ],
    ###
    ]

lst_romeo = [
    ['example-robot-data/robots/romeo_description/urdf/romeo.urdf' ,
     [ ('rg', 'r_gripper', None), ## (name, link_name, tip_link_to_eef)
       ('lg', 'l_gripper', None),
       ('rs', 'r_sole', None),
       ('ls', 'l_sole', None),
      ]
     ],
    ###
    ]

lst_hyq = [
    ['example-robot-data/robots/hyq_description/robots/hyq_no_sensors.urdf' ,
     [ ('lf', 'lf_foot', None), ## (name, link_name, tip_link_to_eef)
       ('rf', 'rf_foot', None),
       ('lh', 'lh_foot', None),
       ('rh', 'rh_foot', None),
      ]
     ],
    ###
    ]

lst_solo = [
    ['example-robot-data/robots/solo_description/robots/solo.urdf' ,
     [ ('fl', 'FL_FOOT', None), ## (name, link_name, tip_link_to_eef)
       ('fr', 'FR_FOOT', None),
       ('hl', 'HL_FOOT', None),
       ('hr', 'HR_FOOT', None),
      ]
     ],
    ###
    ]

lst_anymal_b = [
    ['example-robot-data/robots/anymal_b_simple_description/robots/anymal.urdf' ,
     [ ('lf', 'LF_FOOT', None), ## (name, link_name, tip_link_to_eef)
       ('rf', 'RF_FOOT', None),
       ('lh', 'LH_FOOT', None),
       ('rh', 'RH_FOOT', None),
      ]
     ],
    ###
    ]

lst_anymal_kinova = [
    ['example-robot-data/robots/anymal_b_simple_description/robots/anymal-kinova.urdf' ,
     [ ('lf', 'LF_FOOT', None), ## (name, link_name, tip_link_to_eef)
       ('rf', 'RF_FOOT', None),
       ('lh', 'LH_FOOT', None),
       ('rh', 'RH_FOOT', None),
       ('j1','j2s6s200_end_effector',None),
       ('j2','j2s6s200_end_effector',None),
      ]
     ],
    ###
    ]

lst_panda = [
    ['example-robot-data/robots/panda_description/urdf/panda.urdf' ,
     [ ('pht1', 'panda_hand_tcp', None), ## (name, link_name, tip_link_to_eef)
       ('pht2', 'panda_hand_tcp', None),
      ]
     ],
    ###
    ]

lst_hextilt = [
    ['example-robot-data/robots/hextilt_description/urdf/hextilt_flying_arm_5.urdf' ,
     [ ('gripper_left', 'flying_arm_5__gripper', None), ## (name, link_name, tip_link_to_eef)
       ('gripper_right', 'flying_arm_5__gripper', None), 
     ]
     ],
    ###
    ]

#普通に出たけどエンドエフェクタの座標は出てないｈ
lst_z1 = [
    ['example-robot-data/robots/z1_description/urdf/z1.urdf' ,
     [ ('gripper_up', 'gripperMover', None), ## (name, link_name, tip_link_to_eef)
       ('gripper_dn', 'gripperMover', None), ## (name, link_name, tip_link_to_eef)
      ]
     ],
    ###
    ]

lst_borinot = [
    ['example-robot-data/robots/borinot_description/urdf/borinot_flying_arm_2.urdf' ,
     [ ('arm', 'flying_arm_2__ee', None), ## (name, link_name, tip_link_to_eef)
      ]
     ],
    ###
    ]


lst_kinova = [
    ['example-robot-data/robots/kinova_description/robots/kinova.urdf' ,
     [ ('j1', 'j2s6s200_end_effector', None), ## (name, link_name, tip_link_to_eef)
       ('j2', 'j2s6s200_end_effector', None),
      ]
     ],
    ###
    ]

#テキストファイルは生成されたが空テキスト
lst_bolt = [
    ['example-robot-data/robots/bolt_description/robots/bolt.urdf' ,
     [ ('fl', 'FL_FOOT', None), ## (name, link_name, tip_link_to_eef)
       ('fr', 'FR_FOOT', None),
      ]
     ],
    ###
    ]

lst_anymal_c = [
    ['example-robot-data/robots/anymal_c_simple_description/urdf/anymal.urdf' ,
     [ ('lf', 'LF_FOOT', None), ## (name, link_name, tip_link_to_eef)
       ('rf', 'RF_FOOT', None),
       ('lh', 'LH_FOOT', None),
       ('rh', 'RH_FOOT', None),
      ]
     ],
    ###
    ]

lst_finger_edu = [
    ['example-robot-data/robots/finger_edu_description/robots/finger_edu.urdf' ,
     [ ('ft', 'finger_tip_link', None), ## (name, link_name, tip_link_to_eef)
      
     ]
     ],
    ###
    ]
all_robots = []
all_robots+=lst_a1
all_robots+=lst_b1_z1
all_robots+=lst_b1
all_robots+=lst_go1
all_robots+=lst_romeo_small
all_robots+=lst_romeo
all_robots+=lst_hyq
all_robots+=lst_solo
all_robots+=lst_anymal_b
all_robots+=lst_anymal_kinova
all_robots+=lst_panda
all_robots+=lst_hextilt
all_robots+=lst_kinova
all_robots+=lst_bolt
all_robots+=lst_anymal_c
#18こ




di = DrawInterface()
class EE_Model(RobotModel):
    def __init__(self, urdf_fname, ee_list=None):
        rbt = ib.loadRobotItem(urdf_fname)#ロボットのファイルリンクを読み込む
        super().__init__(rbt)#親クラスであるRobotMelのメソッドにアクセスできる
        if ee_list is not None:#エンドエフェクタのリストがちゃんと渡せている場合
            for ee_info in ee_list:#エンドエフェクタの数だけ
                #ee_list[0]=('fl', 'FL_foot', None)
                #ee_info=略称,正式名称,endeffectorの座標？
                if ee_info[2] is not None:
                    self.registerEndEffector(ee_info[0], ee_info[1],
                                             tip_link_to_eef = ru.make_coordinates(ee_info[2]))
                else :
                    # ここを書きたい
                    link_name = ee_info[1]
                    lk_cds = mkshapes.makeCoords(coords=self.linkCoords(link_name))
                    tmp = self.linkCoords(link_name)\
                    ##画面左はxに対して正、下はzに対して負、、
                    
                    if(urdf_fname=='example-robot-data/robots/anymal_b_simple_description/robots/anymal-kinova.urdf'):
                        if(ee_info[0]=='j1'):
                            tmp.translate(fv(-0.1, 0, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                        elif(ee_info[0]=='j2'):
                            tmp.translate(fv(0.1, 0, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                    
                    elif(urdf_fname=='example-robot-data/robots/panda_description/urdf/panda.urdf'):
                        if(ee_info[0]=='pht1'):
                            tmp.translate(fv(0.01, 0.0, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                        elif(ee_info[0]=='pht2'):
                            tmp.translate(fv(-0.01, 0, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                    elif(urdf_fname=='example-robot-data/robots/kinova_description/robots/kinova.urdf'):
                        if(ee_info[0]=='j1'):
                            tmp.translate(fv(0.1, 0, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                        elif(ee_info[0]=='j2'):
                            tmp.translate(fv(-0.1, 0, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                    
                    elif(urdf_fname=='example-robot-data/robots/b1_description/urdf/b1-z1.urdf'):
                        if(ee_info[0]=='gm_up'):
                            tmp.translate(fv(0.08, 0, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                        elif(ee_info[0]=='gm_dn'):
                            tmp.translate(fv(0.08, 0, -0.02), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                            
                    elif(urdf_fname=='example-robot-data/robots/hextilt_description/urdf/hextilt_flying_arm_5.urdf'):
                        if(ee_info[0]=='gripper_left'):
                            tmp.translate(fv(0, 0.02, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                        elif(ee_info[0]=='gripper_right'):
                            tmp.translate(fv(0, -0.02, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                    elif(urdf_fname=='example-robot-data/robots/z1_description/urdf/z1.urdf'):
                        if(ee_info[0]=='gripper_up'):
                            tmp.translate(fv(0.08, 0, 0), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                        elif(ee_info[0]=='gripper_dn'):
                            tmp.translate(fv(0.08, 0, -0.02), coordinates.wrt.world)## ここのfv(x, y, z)でリンク原点からのオフセットを設定
                            
                    tip_cds = mkshapes.makeCoords(coords=tmp)
                    coords_map_result=ru.make_coords_map(lk_cds.transformation(tip_cds), method='quaternion')
                    #print(ee_info[0], ee_info[1], coords_map_result)
                    self.registerEndEffector(ee_info[0], ee_info[1],
                                             tip_link_to_eef = ru.make_coordinates(coords_map_result))
def setEnvironment(size=512, color=[0, 0, 0], **kwargs):
    ib.setViewSize(size, size)
    ib.setBackgroundColor(color)
    ib.disableGrid()
    ib.setCoordinateAxes(False)

def deleteRobot():
    w=ib.getOrAddWorld()
    w.removeFromParentItem()

def showRobot(on=True):
    w=ib.getOrAddWorld()
    r=w.getChildItem()
    if r is not None:
        r.setChecked(on)

def loadRobot(idx):
    ritm=ib.loadRobotItem(dirname + all_robots[idx][0])
    return RobotModel(ritm)

def set_RandomAngles(robot, sigma=3, **kwargs):
    for j in robot.jointList:
        sig = j.q_upper - j.q_lower
        mu = 0.5*(j.q_upper + j.q_lower)
        if sig > PI:
            sig = PI
        j.q = random.gauss(mu, sig/(2*sigma))
    robot.hook()

    
def setRandomCamera():
    theta = random.gauss(0, PI/2)
    elv   = random.gauss(0, PI/4)
    ar = IC.normalizeVector(fv(math.sin(theta), math.cos(theta), math.tan(elv)))
    #ln = mkshapes.makeLines([[0, 0, 0], ar.tolist()])
    #di.addObject(ln)
    cds=ib.cameraPositionLookingAt(ar, fv(0, 0, 0), fv(0, 0, 1))
    ib.setCameraCoords(cds)
    ib.viewAll()

##
def saveImage(prefix, *args):
    ib.viewAll()
    filename = prefix.format(*args)
    ib.saveImageOfScene(filename)

def saveLabel_joint(prefix, *args, pointSize=10, robot=None):
    filename = prefix.format(*args)
    di.clear()
    lst= []
    for j in robot.jointList:
        if j.jointTypeLabel == 'revolute':
            lst.append( j.getCoords().pos )
    ### get end-effector position
    pt = mkshapes.makePoints(lst, pointSize=pointSize)
    di.addObject(pt)
    ib.saveImageOfScene(filename)
    di.clear()
    
def saveLabel_endeffector(prefix, *args, pointSize=10, robot=None):
    filename = prefix.format(*args)
    di.clear()
    lst=[]
    ### get end-effector position
    for k, v in robot.eef_map.items():
        ee_cds = v.endEffector
        lst.append( ee_cds.pos )
        #print(ee_cds)
    pt = mkshapes.makePoints(lst, pointSize=pointSize)
    di.addObject(pt)
    ib.saveImageOfScene(filename)
    di.clear()
#setEnvironment()
def binarize_to_fill_image(image):#塗りつぶし関数
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 白黒を反転
    inverted = cv2.bitwise_not(gray)

    # 二値化（閾値127を基準に黒か白にする）
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    # 輪郭を検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 同じサイズの空の画像を作成
    filled_image = np.zeros_like(image)
    # 輪郭内を塗りつぶす
    for contour in contours:
        # 輪郭の内部を塗りつぶす
        cv2.drawContours(filled_image, [contour], -1, (255,255,255), thickness=cv2.FILLED)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 輪郭を描画するために全て黒の画像を作成
    input_image = np.zeros_like(binary)
    
    # 検出した輪郭を塗りつぶし
    cv2.drawContours(input_image, contours, -1, (255),2)
    return filled_image,input_image
def CreateOutlineImage(image_path,output_path):
# 2. 画像を読み込み
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Image at path {image_path} は読み込めませんでした。")
        return


    # 画像がモノクロかカラーかで処理を分ける
    if len(image_cv.shape) == 2:  # モノクロ画像の場合（高さ, 幅）
        image_cv = np.expand_dims(image_cv, axis=-1)  # チャンネル次元を追加
    elif len(image_cv.shape) == 3:  # カラー画像の場合（高さ, 幅, チャンネル）
        if image_cv.shape[2] != 3:
            raise ValueError(f"Unexpected number of channels: {image_cv.shape[2]}")
    device = 'cuda'
    # 画像を(チャンネル, 高さ, 幅)に変換
    image = np.transpose(image_cv, (2, 0, 1)).astype(np.float32) / 255
    image = image[np.newaxis, :, :, :]
    image = torch.Tensor(image).to(device)

    # 3. 頭マップを作成
    headmap_np = image.sigmoid().cpu().detach().numpy()[0, 0]
    gray = (headmap_np * 255).astype(np.uint8)

    # 4. モード値を計算
    mode_result = stats.mode(gray.flatten(), keepdims=False)  # モードを計算
    mode_value = float(mode_result[0])  # スカラー値として扱うために float に変換
    #print(f"Background mode value: {mode_value}")

    # 5. 二値化（モードをしきい値として使用）
    _, binary = cv2.threshold(gray, mode_value, 255, cv2.THRESH_BINARY)

    # 6. 二値画像をnp.uint8に変換
    binary = binary.astype(np.uint8)

    # 7. 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 8. 黒い背景の画像を作成
    height, width = image_cv.shape[:2]  # 元の画像と同じサイズの黒画像
    black_background = np.zeros((height, width, 3), dtype=np.uint8)

    # 9. 黒い背景に輪郭を描画
    cv2.drawContours(black_background, contours, -1, (255, 255, 255), 2)  # 白色で輪郭を描画

    # 10. 結果を保存（画像名にiを付けて保存）
    #cv2.imwrite(os.path.join(output_dir, f'contour_on_black_gray_{str(i).zfill(8)}.png'), gray)
    #cv2.imwrite(os.path.join(output_dir, f'contour_on_black_binary_{str(i).zfill(8)}.png'), binary)
    cv2.imwrite(output_path, black_background)


    
#いまわかんないのはこのプログラムの使い方(mainが関数になっているからどこでどの引数でつかうのか)
#robot.joint_Listのようにエンドエフェクタのlistがあるのか
def make_dataset(total_num=30000, **kwargs):
    setEnvironment(**kwargs)
    total_counter = 0
    root_path = '/dataset'
    #csvは相対パスにしてあるので、おく場所はrootの中
    dir_path = os.path.join(root_path, 'joint_dataset')
    absolute_path = os.path.abspath(dir_path)
    os.makedirs(os.path.join(absolute_path, 'image'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'label'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'label/endeffector'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'label/joint'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'outline'), exist_ok=True)
    
    print("絶対パス:", absolute_path)
    
    for l in all_robots:
        deleteRobot()
        robot = EE_Model(l[0], l[1])
        print(l[0])
        for i in range(total_num//(len(all_robots)-1)): ## 1600
            set_RandomAngles(robot, **kwargs)
            setRandomCamera()
            showRobot(True)
            saveImage('{0}/image_{1:08d}.png', os.path.join(absolute_path, 'image'), total_counter)
            print(os.path.join(absolute_path, 'image'))
            #↑で保存したimageを早速読み込み、アウトラインにして保存する関数
            image_path = f'/dataset/joint_dataset/image/image_{total_counter:08d}.png'
            output_path_outline = f'/dataset/joint_dataset/outline/outline_{total_counter:08d}.png'
            print("読み込み画像パス:", image_path)
            CreateOutlineImage(image_path, output_path_outline)
            
            showRobot(False)
            saveLabel_joint('{0}/label_joint_{1:08d}.png', os.path.join(absolute_path, 'label/joint'), total_counter, robot=robot)
            saveLabel_endeffector('{0}/label_endeffector_{1:08d}.png', os.path.join(absolute_path, 'label/endeffector'), total_counter, robot=robot)
    
            # 3つの画像を読み込んで確認、リサイズ、結合
            img1 = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img2 = cv2.imread(f'/dataset/joint_dataset/label/joint/label_joint_{total_counter:08d}.png', cv2.IMREAD_COLOR)
            img3 = cv2.imread(f'/dataset/joint_dataset/label/endeffector/label_endeffector_{total_counter:08d}.png', cv2.IMREAD_COLOR)
            
            # 各画像のサイズを確認してリサイズする
            target_shape = img1.shape[:2]  # 高さと幅のターゲットサイズを取得
            if img2.shape[:2] != target_shape:
                img2 = cv2.resize(img2, (target_shape[1], target_shape[0]))
            if img3.shape[:2] != target_shape:
                img3 = cv2.resize(img3, (target_shape[1], target_shape[0]))
            
            # 各画像のデータ型を確認して変換
            dtype = img1.dtype
            if img2.dtype != dtype:
                img2 = img2.astype(dtype)
            if img3.dtype != dtype:
                img3 = img3.astype(dtype)
            
            # 画像を結合して保存
            show_img = cv2.hconcat([img1, img2, img3])
            output_path_all = f'/dataset/joint_dataset/all/all_{total_counter:08d}.png'
            cv2.imwrite(output_path_all, show_img)
    
            total_counter += 1


def draw_true_image(image,true_joint_points,true_end_points):#正解がある場合、正解の描画
        #jointを青で描画
        joint_points = true_joint_points[:, ::-1]
        end_points = true_end_points[:, ::-1]
        for point in joint_points:
            cv2.circle(image, tuple(point), 2, (255, 0, 0), -1)  # (BGR)

        # オブジェクトポイントを青で描画
        for point in end_points:
            cv2.circle(image, tuple(point), 2, (0, 0, 255), -1)  # (BGR)

        return image

#ジョイントの情報をcsvに追加する
import csv
import os
import numpy as np

def export_joint_and_eef_info_to_csv(csv_path, image_filename, robot, write_header=False):
    """
    ジョイント情報およびエンドエフェクタ情報をCSVに書き込む。

    Parameters:
    - csv_path: 出力先CSVファイルのパス
    - image_filename: 対応する画像ファイル名（例：'image_00000001.png'）
    - robot: EE_Model型のロボットインスタンス
    - write_header: Trueの場合、ヘッダーを書き込む（通常は最初だけTrue）
    """
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow([
                "image_name",
                "type",             # 'joint' または 'eef'
                "name",             # ジョイント名 or エンドエフェクタ名
                "joint_type",       # ジョイントの場合の型（revolute など） / eefなら空欄
                "axis",             # 回転軸（x/y/z）/ eefなら空欄
                "q_angle(rad)",     # 現在の角度 / eefなら空欄
                "q_lower(rad)",     # 下限 / eefなら空欄
                "q_upper(rad)",     # 上限 / eefなら空欄
                "position_x",       # 座標
                "position_y",
                "position_z"
            ])

        # --- ジョイント情報 ---
        for joint in robot.jointList:
            joint_name = joint.name
            joint_type = joint.jointTypeLabel
            pos = joint.getCoords().pos
            axis_local = joint.jointAxis

            if np.allclose(axis_local, [1, 0, 0]):
                axis_label = "x"
            elif np.allclose(axis_local, [0, 1, 0]):
                axis_label = "y"
            elif np.allclose(axis_local, [0, 0, 1]):
                axis_label = "z"
            else:
                axis_label = str(axis_local.tolist())

            writer.writerow([
                image_filename,
                "joint",
                joint_name,
                joint_type,
                axis_label,
                joint.q,
                joint.q_lower,
                joint.q_upper,
                pos[0], pos[1], pos[2]
            ])

        # --- エンドエフェクタ情報 ---
        for eef_name, eef in robot.eef_map.items():
            pos = eef.endEffector.pos
            writer.writerow([
                image_filename,
                "eef",
                eef_name,
                "", "", "", "", "",  # ジョイント用カラムは空欄
                pos[0], pos[1], pos[2]
            ])


def make_dataset_test5(total_num=30, **kwargs):
    setEnvironment(**kwargs)
    total_counter = 0
    root_path = 'dataset'
    #csvは相対パスにしてあるので、おく場所はrootの中
    dir_path = os.path.join(root_path, 'joint_dataset_test5')
    absolute_path = os.path.abspath(dir_path)
    os.makedirs(os.path.join(absolute_path, 'image'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'label'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'label/endeffector'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'label/joint'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'outline'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'all'), exist_ok=True)
    os.makedirs(os.path.join(absolute_path, 'true'), exist_ok=True)
    a='dataset/joint_dataset_test5/image'
    print("絶対パス:", absolute_path)
    if os.path.exists(a):
        print("ファイルまたはディレクトリが存在します",a)
    else:
        print("存在しません")
    for l in all_robots:
        deleteRobot()
        print("aaaaaaaaaaaaaaaaa")
        print(l[0],l[1])
        urdf_path = os.path.join(dirname, l[0])  # dirnameが''ならl[0]そのまま

        print("---- パス確認 ----")
        print("URDFパス:", urdf_path)
    
        if os.path.exists(urdf_path):
            print("✅ 存在します:", urdf_path)
        else:
            print("❌ 存在しません！:", urdf_path)
        
        robot = EE_Model(l[0], l[1])
        for i in range(total_num//(len(all_robots)-1)): ## 1600
            set_RandomAngles(robot, **kwargs)
            setRandomCamera()
            showRobot(True)
            saveImage('{0}/image_{1:08d}.png', os.path.join(absolute_path, 'image'), total_counter)

            # === CSV出力処理 ===
            csv_path = os.path.join(absolute_path, "joint_and_eef_info.csv")
            
            # 最初だけヘッダーを書く
            write_header = (total_counter == 0 and not os.path.exists(csv_path))
            
            # 対応画像ファイル名
            img_filename = f"image_{total_counter:08d}.png"
            
            # CSVにジョイント＋エンドエフェクタ情報を出力
            export_joint_and_eef_info_to_csv(
                csv_path=csv_path,
                image_filename=img_filename,
                robot=robot,
                write_header=write_header
            )



            print(os.path.join(absolute_path, 'image'))
            #↑で保存したimageを早速読み込み、アウトラインにして保存する関数
            image_path = f'dataset/joint_dataset_test5/image/image_{total_counter:08d}.png'
            output_path_outline = f'dataset/joint_dataset_test5/outline/outline_{total_counter:08d}.png'
            print("読み込み画像パス:", image_path)
            CreateOutlineImage(image_path, output_path_outline)
            
            showRobot(False)
            saveLabel_joint('{0}/label_joint_{1:08d}.png', os.path.join(absolute_path, 'label/joint'), total_counter, robot=robot)
            saveLabel_endeffector('{0}/label_endeffector_{1:08d}.png', os.path.join(absolute_path, 'label/endeffector'), total_counter, robot=robot)
    
            # 3つの画像を読み込んで確認、リサイズ、結合
            img1 = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img2 = cv2.imread(f'dataset/joint_dataset_test5/label/joint/label_joint_{total_counter:08d}.png', cv2.IMREAD_COLOR)
            img3 = cv2.imread(f'dataset/joint_dataset_test5/label/endeffector/label_endeffector_{total_counter:08d}.png', cv2.IMREAD_COLOR)
            true_joint=cv2.imread(f'dataset/joint_dataset_test5/label/joint/label_joint_{total_counter:08d}.png')
            true_end=cv2.imread(f'dataset/joint_dataset_test5/label/endeffector/label_endeffector_{total_counter:08d}.png')        
            true_joint_points = np.argwhere(np.any(true_joint != 0, axis=-1))
            true_end_points = np.argwhere(np.any(true_end != 0, axis=-1))
            true_image=draw_true_image(img1,true_joint_points,true_end_points)
            cv2.imwrite(f'dataset/joint_dataset_test5/true/true_{total_counter:08d}.png',true_image)
            # 各画像のサイズを確認してリサイズする
            target_shape = img1.shape[:2]  # 高さと幅のターゲットサイズを取得
            if img2.shape[:2] != target_shape:
                img2 = cv2.resize(img2, (target_shape[1], target_shape[0]))
            if img3.shape[:2] != target_shape:
                img3 = cv2.resize(img3, (target_shape[1], target_shape[0]))
            
            # 各画像のデータ型を確認して変換
            dtype = img1.dtype
            if img2.dtype != dtype:
                img2 = img2.astype(dtype)
            if img3.dtype != dtype:
                img3 = img3.astype(dtype)
            
            # 画像を結合して保存
            show_img = cv2.hconcat([img1, img2, img3])
            output_path_all = f'dataset/joint_dataset_test5/all/all_{total_counter:08d}.png'
            success=cv2.imwrite(output_path_all, show_img)
            # 保存が成功したか確認
            if success:
                print(f"画像が正常に保存されました: {output_path_all}")
            else:
                print("画像の保存に失敗しました")
            total_counter += 1
            #print("絶対パス:", absolute_path)
        if os.path.exists('dataset/joint_dataset_test5/image/image_00000000.png'):
            print("ファイルまたはディレクトリが存在します",a)