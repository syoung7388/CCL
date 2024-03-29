
echo python3 do_eval.py --dataset './datasets/slurp' --target 'google' --our_model './best_models/slurp_google.pt' 
python3 do_eval.py --dataset './datasets/slurp' --target 'google' --our_model './best_models/slurp_google.pt' --batch_size 64

echo python3 do_eval.py --dataset './datasets/slurp' --target 'wave2vec2.0' --our_model './best_models/slurp_wave2vec2.0.pt'
python3 do_eval.py --dataset './datasets/slurp' --target 'wave2vec2.0' --our_model './best_models/slurp_wave2vec2.0.pt' --batch_size 64


echo python3 do_eval.py --dataset './datasets/timers' --target 'google' --our_model './best_models/timers_google.pt' 
python3 do_eval.py --dataset './datasets/timers' --target 'google' --our_model './best_models/timers_google.pt' --batch_size 64

echo python3 do_eval.py --dataset './datasets/timers' --target 'wave2vec2.0' --our_model './best_models/timers_wave2vec2.0.pt'
python3 do_eval.py --dataset './datasets/timers' --target 'wave2vec2.0' --our_model './best_models/timers_wave2vec2.0.pt' --batch_size 64


echo python3 do_eval.py --dataset './datasets/fsc' --target 'google' --our_model './best_models/fsc_google.pt' 
python3 do_eval.py --dataset './datasets/fsc' --target 'google' --our_model './best_models/fsc_google.pt' --batch_size 64

echo python3 do_eval.py --dataset './datasets/fsc' --target 'wave2vec2.0' --our_model './best_models/fsc_wave2vec2.0.pt'
python3 do_eval.py --dataset './datasets/fsc' --target 'wave2vec2.0' --our_model './best_models/fsc_wave2vec2.0.pt' --batch_size 64

echo python3 do_eval.py --dataset './datasets/snips' --target 'wave2vec2.0' --our_model './best_models/snips_wave2vec2.0.pt' 
python3 do_eval.py --dataset './datasets/snips' --target 'wave2vec2.0' --our_model './best_models/snips_wave2vec2.0.pt' --batch_size 64
