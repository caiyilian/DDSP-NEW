import os
import re
import subprocess
from tqdm import tqdm
from os.path import basename, dirname
Dict = {}

for log_dir in ["/public/cyl/DDSP-SSECP/checkpoint/all_merge_pseudo_pro12", "/public/cyl/DDSP-SSECP/checkpoint/mmwhs"]:
    # log_dir = "/public/cyl/DDSP-SSECP/checkpoint/all_merge_pseudo_pro12"

    for each_path in tqdm(os.listdir(log_dir)):
        check_path = os.path.join(log_dir, each_path)
        check_path = os.path.join(check_path, os.listdir(check_path)[0])
        direction = basename(dirname(check_path)).split("_")[-1]
        fold = basename(check_path).split("_")[-1]
        if basename(dirname(dirname(check_path))) == 'all_merge_pseudo_pro12':
            num_classes = 2
            A_root = "./Pro128/BIDMC"
            B_root = "./Pro128/HK"
            dataset = 'pro'
        else:
            num_classes = 5
            A_root = "./mmwhs_96/ct_debugging96"
            B_root = "./mmwhs_96/mr_debugging96"
            dataset = 'mmwhs'
        if Dict.get(dataset) is None:
            Dict[dataset] = {}
        if Dict[dataset].get(direction) is None:
            Dict[dataset][direction] = {}
        if Dict[dataset][direction].get(fold) is None:
            Dict[dataset][direction][fold] = []
        if os.path.exists(os.path.join(check_path, "testing_dice.txt")):
            with open(os.path.join(check_path, "testing_dice.txt"),'r') as f:
                testing_dice = float(f.read())
            Dict[dataset][direction][fold].append(testing_dice)
        else:
            print("没有找到cache文件，所以重新运行")
            # 使用 subprocess 获取输出
            cmd = f"python test_pro.py --fold {fold} --direction {direction} --num_classes {num_classes} --checkpoint {check_path} --A_root {A_root} --B_root {B_root}"

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            # 获取标准输出
            output = result.stdout
            # 使用正则表达式提取 Testing dice 的值
            match = re.search(r'Testing dice\s*:\s*([\d.]+)%', output)
            if match:
                testing_dice = float(match.group(1))
                # print(f"Testing dice:  {testing_dice}")
                Dict[dataset][direction][fold].append(testing_dice)
                with open(os.path.join(check_path, "testing_dice.txt"),'w') as f:
                    f.write(str(testing_dice))
            else:
                print("未找到 Testing dice")

direction_dict = {}
for dataset in Dict.keys():
    for direction in Dict[dataset].keys():
        if direction_dict.get(f"{dataset}-{direction}") is None:
            direction_dict[f"{dataset}-{direction}"] = []
        for fold in Dict[dataset][direction].keys():
            print(f"{dataset}-{direction}-{fold}:{Dict[dataset][direction][fold]},Max:{max(Dict[dataset][direction][fold])}")
            direction_dict[f"{dataset}-{direction}"].append(max(Dict[dataset][direction][fold]))
print("--------------每个数据集指标-------------")
for key in direction_dict.keys():
    print(f"{key}:{sum(direction_dict[key])/len(direction_dict[key])}")
    
"""
pro-A2B:87.03    (83.95) ok
pro-B2A:84.745   (84.48) ok
mmwhs-B2A:91.89  (92.09)
mmwhs-A2B:85.94  (85.59) ok

"""
        