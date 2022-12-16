#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=12:00:00
#SBATCH --mem=100000
#SBATCH --job-name=pp1953
#SBATCH --mail-user=pp1953p@nyu.edu
#SBATCH --output=output/slurm_%j.out


. ~/.bashrc
module load anaconda3/5.3.1

conda activate PPUU
conda install -n PPUU nb_conda_kernels

chikka=$1

echo $chikka


cd 
cd /home/pp1953/code/official
# python hyper_supervise_validation.py --dataset="prid_subset" >> ~/code/Personre-id-master/outputs/prid_diff_val_map.out
# python hyper_supervise_validation.py  --dataset="ilidsvid_subset" >> ~/code/Personre-id-master/outputs/ilvid_subset_rand_bt6.out
# python hyper_supervise_validation.py  --focus="map" --dataset="mars_subset" --sampling="intelligent" >> ~/code/temp/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them-master/output/intelligent_mars_cl_centers.out
# python hyper_supervise_validation.py  --focus="map" --dataset="mars_subset2" --sampling="intelligent" >> ~/code/official/output/subset2_intelligent_mars_cl_centers.out
# python hyper_supervise_validation.py  --focus="map" --dataset="mars_subset2"  >> ~/code/official/output/subset2_random_mars_cl_centers.out


# python config_trainer.py --focus=map --dataset=mars --opt=$chikka --name=_mars_cl_centers_ --cl-centers >>  ~/code/official/output/mars_cl_centers_$chikka.out
# python config_trainer.py --focus=map --dataset=mars --opt=$chikka --name=_mars_attncl_centers_ --cl-centers --attn-loss >>  output/2mars_attn_cl_centers_$chikka.out
# python3 config_trainer.py --focus=map --dataset=mars --opt=$chikka --name=_mars_osm_ --use-OSMCAA >>  output/mars_osm_$chikka.out
# python3 config_trainer.py --focus=map --dataset=mars --opt=$chikka --name=_mars_osm_ --use-OSMCAA >>  output/mars_osm_$chikka.out
python config_trainer.py --focus=map --dataset=mars --opt=$chikka --name=_mars_cl_centers_ --cl-centers >>  ~/code/official/output/mars_cl_centers_$chikka.out
# python config_trainer.py --focus=map --dataset=prid --opt=$chikka --name=_prid_cl_centers_ --cl-centers >>  ~/code/official/output/prid_cl_centers_$chikka.out
# python3 config_trainer.py --focus=map --dataset=mars --opt=2 --name=_mars_cl_centers_ --cl-centers >> output/mars_cl_centers_2.out

# python bagoftricks.py --pretrained-model="/beegfs/pp1953/ResNet50ta_bt_mars_cl_centers__8__checkpoint_ep181.pth.tar" -d="prid" --opt=$chikka --name="_prid_CL_CENTERS_" --validation-training --cl-centers --print-freq=10 
# python3 config_trainer.py --focus=map --dataset=prid --opt=38 --name=_prid_cl_centers_ --cl-centers >> output/prid_cl_centers_38.out