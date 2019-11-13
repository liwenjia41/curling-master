## Curling Detection

### Setup：

- [Installation](https://github.com/liwenjia41/curling-master/blob/master/research/object_detection/g3doc/installation.md)

### Samples Prepare:

- Download existing dataset
  - [BaiduYun](https://pan.baidu.com/s/1TWD2H2ELDl7cqHAh9PU7eQ)
  - [BaiduYun] (Later...)

- Generate your own dataset
  
- You can use [LabelImg](https://github.com/tzutalin/labelImg) to label your sample
  
- (Optional) Image Augment

  - You can use your ways to augment images or

  - Use `preprocess.py`

    ```
    python preprocess.py --img_dir=/path/to/image/dir --direct=train\val
    ```

- Convert xml to csv

  ```text
  python xml_to_csv.py --input_path=/path/to/dir/save/xml --output=xxx.csv
  ```

- Generate tfrecord for tensorflow model

  ```
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record --dir=images
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=test.record --dir=images
  ```

### Training:

- Download [pretrained models](https://github.com/liwenjia41/curling-master/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
  - Put models in your directory (eg. my_pretrained_model)

- Prepare label_map.pbtxt

  - You can copy [pet_label_map.pbtxt](https://github.com/liwenjia41/curling-master/blob/master/research/object_detection/data/pet_label_map.pbtxt) in this folder and then change:

    ```
    item {
    	id: 1
    	name: 'curling'
    }
    ```

- Modify .config file

  - In the [object_detection/samples/configs](https://github.com/liwenjia41/curling-master/tree/master/research/object_detection/samples/configs) folder, there are skeleton object_detection configuration files.Select the config file corresponding to the network model and copy it in this folder.

  - For examples, `ssdlite_mobilenet_v2.coco.config` .Change .config file in these place:

    ```
    num_classes: 90  ==>  num_classes: 1(only curling)
    
    
    fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"  ==>  fine_tune_checkpoint: "my_pretrained_model/model.ckpt"
    
    
    (Optional)
    # data_augmentation_options {
    # 	random_horizontal_flip {
    # 	}
    # }
      			
            
    num_steps: 200000  ==>  (whatever you want)
      		
          
    train_input_reader: {
      tf_record_input_reader {
        input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record-?????-of-00100"
      }
      label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
    }  
    ==>
    train_input_reader: {
      tf_record_input_reader {
        input_path: "/path/to/train.record"
      }
      label_map_path: "/path/to/label_map.pbtxt"
    }
    
    
    eval_input_reader: {
      tf_record_input_reader {
        input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record-?????-of-00010"
      }
      label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
      shuffle: false
      num_readers: 1
    } 
    ==>
    eval_input_reader: {
      tf_record_input_reader {
        input_path: "/path/to/test.record"
      }
      label_map_path: "/path/to/label_map.pbtxt"
      shuffle: false
      num_readers: 1
    }
    ```

- Train

  ```
  python object_detection/legacy/train.py 
  				--logtostderr
  				--train_dir=/path/to/train_dir(New folder created by yourself)
  				--pipeline_config_path=/path/to/.config
  ```

  - After training, the model would be saves in your train_dir

- [Exporting a trained model for inference](https://github.com/liwenjia41/curling-master/blob/master/research/object_detection/g3doc/exporting_models.md)



### Test:

```
# detect one image using trained model:
python detect_with_ssd.py --model=/train_model/xxx.pb --label=xxx_label_map.pbtxt --mode=image --path=/path/to/image
  
# detect video using trained model:
python detect_with_ssd.py --model=/train_model/xxx.pb --label=xxx_label_map.pbtxt --mode=video --path=/path/to/video
```

