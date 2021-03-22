#!/bin/bash
#SBATCH --job-name=ft         # nom du job
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --qos=qos_gpu-t3                # qos: quality of service
#SBATCH --gres=gpu:4         # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=40          # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --exclusive
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=6:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/msk/urt22vb/JustAsk/ft_logs/%A_%a.out     # nom du fichier de sortie
#SBATCH --error=/gpfsdswork/projects/rech/msk/urt22vb/JustAsk/ft_logs/%A_%a.err      # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-4
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
mlm_prob=0.15
datasets=(msrvtt activitynet msvd ivqa how2qa)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
lrs=(0.00001 0.00002 0.00003 0.00001 0.00005)
lr=${lrs[$SLURM_ARRAY_TASK_ID]}
wds=(0 0.0001 0 0 0)
weight_decay=${wds[$SLURM_ARRAY_TASK_ID]}
pretrain_path="/gpfsstore/rech/msk/urt22vb/cvpr21/checkpoints/ptsqa/e9.pth"
checkpoint_predir="/gpfsstore/rech/msk/urt22vb/cvpr21/checkpoints/ptsqae9_ft/"
batch_size_val=2048
batch_size=256

mkdir -p $checkpoint_predir
/gpfswork/rech/msk/urt22vb/miniconda3/bin/python /gpfswork/rech/msk/urt22vb/JustAsk/main_videoqa.py --num_thread_reader=16 --checkpoint_dir=$SLURM_ARRAY_TASK_ID --mlm_prob=$mlm_prob --dataset=$dataset --lr=$lr --pretrain_path=$pretrain_path --checkpoint_predir=$checkpoint_predir --batch_size_val=$batch_size_val --batch_size=$batch_size
