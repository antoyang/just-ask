#!/bin/bash
#SBATCH --job-name=pt_sqa         # nom du job
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --partition=gpu_p2                  # queue
#SBATCH --qos=qos_gpu-t4                # qos: quality of service
#SBATCH --gres=gpu:8         # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=24          # nombre de coeurs à réserver (un quart du noeud)
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=48:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/msk/urt22vb/JustAsk/logs/sqa_%j.out     # nom du fichier de sortie
#SBATCH --error=/gpfsdswork/projects/rech/msk/urt22vb/JustAsk/logs/sqa_%j.err      # nom du fichier d'erreur (ici commun avec la sortie)
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
checkpoint_dir="ptsqa"
qmax_words=20
amax_words=10
max_feats=20
batch_size=128
batch_size_val=256
n_pair=32
freq_display=20
lr=0.00005
mlm_prob=0.15

/gpfswork/rech/msk/urt22vb/miniconda3/bin/python /gpfswork/rech/msk/urt22vb/JustAsk/main_sqa.py --dataset="sqa" --num_thread_reader=16 --epochs=10 --checkpoint_dir=$checkpoint_dir --qmax_words=$qmax_words --amax_words=$amax_words --max_feats=$max_feats --batch_size=$batch_size --batch_size_val=$batch_size_val --n_pair=$n_pair --freq_display=$freq_display --lr=$lr --mlm_prob=$mlm_prob