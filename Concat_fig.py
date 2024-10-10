from PIL import Image
import os

def combine_images(folder1, folder2, output_folder):
    for filename in os.listdir(folder1):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path1 = os.path.join(folder1, filename)
            file_path2 = os.path.join(folder2, filename)
            output_path = os.path.join(output_folder, filename)
            
            img1 = Image.open(file_path1)
            img2 = Image.open(file_path2)

            width = img1.width + img2.width
            height = max(img1.height, img2.height)

            new_image = Image.new("RGB", (width, height))
            new_image.paste(img1, (0, 0))
            new_image.paste(img2, (img1.width, 0))
            
            new_image.save(output_path)

if __name__ == '__main__':
    #对比motrv2与transdetr
    # folder = "/home/ubuntu/TransDETR-original/exps/e2e_TransVTS_r50_ICDAR15/eval/results" # trans_detr
    # seq_nums = []
    # for seq in os.listdir(folder):
    #     folder1 = "/home/ubuntu/TransDETR-original/exps/e2e_TransVTS_r50_ICDAR15/eval/results/" +seq
    #     folder2 = "/home/ubuntu/MOTRv2-trans/result/eval/img/"+ seq # motrV + rotate 
    #     output_folder = "/home/ubuntu/MOTRv2-trans/contrast/" + seq
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)

    #     combine_images(folder1, folder2, output_folder)

    # print("生成完成")
#
    folder = "/home/ubuntu/MOTRv2-trans/result_阶段1/eval_DS_ori_0.5/img/" # trans_detr
    seq_nums = []
    for seq in os.listdir(folder):
        folder1 = "/home/ubuntu/MOTRv2-trans/result_阶段1/eval_DS_ori_0.5/img/" +seq
        folder2 = "/home/ubuntu/MOTRv2-trans/result/eval_nochange_loss_changeProposal_bigdata_epoch14_lrDrop_5******/img/"+ seq # motrV + rotate 
        output_folder = "/home/ubuntu/MOTRv2-trans/Contrast_iou/oriAndPaigeNet/" + seq
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        combine_images(folder1, folder2, output_folder)

    print("生成完成")
    