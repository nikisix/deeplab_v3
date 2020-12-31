#!/bin/bash
# workon summ
# python deeplab_saved_model.py --flagfile=flags.conf
rm -r versions/2
python bytes_deeplab_saved_model.py --flagfile=flags.conf

# OUTPUT
# INFO:tensorflow:No assets to save.
# I1210 17:14:49.093361 4513277376 builder_impl.py:640] No assets to save.
# INFO:tensorflow:No assets to write.
# I1210 17:14:49.093558 4513277376 builder_impl.py:460] No assets to write.
# INFO:tensorflow:SavedModel written to: ./versions/1/saved_model.pbtxt
# I1210 17:14:50.771157 4513277376 builder_impl.py:425] SavedModel written to: ./versions/1/saved_model.pbtxt
# Done exporting!

# can inspect with:
# workon tf
# saved_model_cli show --dir 2 --all
