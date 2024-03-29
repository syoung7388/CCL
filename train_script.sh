

python3 contrastive_learning.py --dataset './datasets/slurp' --target google --ckpt slurp_google
python3 consistency_learning.py --dataset './datasets/slurp' --target google --ckpt slurp_google --lr 1e-5 --lambda1 0.0 --lambda2 1.0 --tlm_path '/slurp_google/models/tlm/e=5.pt' --ilm_path '/slurp_google/models/ilm/e=5.pt'

python3 contrastive_learning.py --dataset './datasets/slurp' --target wave2vec2.0 --ckpt slurp_wave2vec2.0
python3 consistency_learning.py --dataset './datasets/slurp' --target wave2vec2.0 --ckpt slurp_wave2vec2.0 --lr 1e-4 --lambda1 0.3 --lambda2 0.7 --tlm_path '/slurp_wave2vec2.0/models/tlm/e=5.pt' --ilm_path '/slurp_wave2vec2.0/models/ilm/e=5.pt'

python3 contrastive_learning.py --dataset './datasets/timers' --target google --ckpt timers_google
python3 consistency_learning.py --dataset './datasets/timers' --target google --ckpt timers_google --lr 1e-5 --lambda1 0.5 --lambda2 0.5 --tlm_path '/timers_google/models/tlm/e=5.pt' --ilm_path '/timers_google/models/ilm/e=5.pt'

python3 contrastive_learning.py --dataset './datasets/timers' --target wave2vec2.0 --ckpt timers_wave2vec2.0
python3 consistency_learning.py --dataset './datasets/timers' --target wave2vec2.0 --ckpt timers_wave2vec2.0 --lr 1e-5 --lambda1 0.4 --lambda2 0.6 --tlm_path '/timers_wave2vec2.0/models/tlm/e=5.pt' --ilm_path '/timers_wave2vec2.0/models/ilm/e=5.pt'


python3 contrastive_learning.py --dataset './datasets/fsc' --target google --ckpt fsc_google
python3 consistency_learning.py --dataset './datasets/fsc' --target google --ckpt fsc_google --lr 5e-5 --lambda1 0.1 --lambda2 0.9 --tlm_path '/fsc_google/models/tlm/e=5.pt' --ilm_path '/fsc_google/models/ilm/e=5.pt'

python3 contrastive_learning.py --dataset './datasets/fsc' --target wave2vec2.0 --ckpt fsc_wave2vec2.0
python3 consistency_learning.py --dataset './datasets/fsc' --target wave2vec2.0 --ckpt fsc_wave2vec2.0 --lr 5e-5 --lambda1 0.0 --lambda2 1.0 --tlm_path '/fsc_wave2vec2.0/models/tlm/e=5.pt' --ilm_path '/fsc_wave2vec2.0/models/ilm/e=5.pt'

python3 contrastive_learning.py --dataset './datasets/snips' --target wave2vec2.0 --ckpt snips_wave2vec2.0
python3 consistency_learning.py --dataset './datasets/snips' --target wave2vec2.0 --ckpt snips_wave2vec2.0 --lr 3e-5 --lambda1 0.9 --lambda2 0.1 --tlm_path '/snips_wave2vec2.0/models/tlm/e=5.pt' --ilm_path '/snips_wave2vec2.0/models/ilm/e=5.pt'


