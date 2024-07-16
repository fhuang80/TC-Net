import os

experiment_list = ['230314_v7_routetype1_missingtype3_Tree', '230314_v7_routetype1_missingtype3_Tree_pretrain_0']

for i in range(len(experiment_list)):
    for epoch in range(0, 210, 10):
        cmd = 'python show_recon.py --data_root ../../Data/OwnTree/ --batchSize 1 --distributed --visual ' \
              + ' --route_type 4 --missing_type 3 --netG ../../Predict/OwnTree/' + experiment_list[i] + '/point_netG' + str(epoch) +'.pth'
        print('experiment_dir:', experiment_list[i], 'epoch: ', str(epoch))
        os.system(cmd)
    os.system('@mkdir ' + experiment_list[i])
    os.system('@move ' + '*.png ' + experiment_list[i])