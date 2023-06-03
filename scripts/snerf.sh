export CUDA_VISIBLE_DEVICES=4
#python3 -m run --ginc configs/snerf/llff_render.gin --scene_name Glass
python3 -m run --ginc configs/snerf/llff.gin --scene_name Glass
