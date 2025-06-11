'''
This is to get a dataloader for each domain dataset (k number)

'''
from merging.merges.dataset_merge.data_factory_seq import CV_data

def Load_dataset(argw):
    
    data_ = CV_data(                     
                            path = argw["data_path"],
                            sub_dataset = argw["sub_dataset"],
                            permute_ = argw["permute"],
                            sqs = argw["sqw"],
                            batch_training = argw["batch_training"],
                            batch_testing = argw["batch_testing"],
                            num_workers = 10,
                            
                            transform_set = {"random_crop": argw["rc"],
                                            "autoaug": argw["aa"],
                                            "random_crop_pase": argw["rcp"],
                                            "resize": argw["rs"]},
                            multitasking = argw["multitasking"],
                            train_test= argw["train_test"],
                            fix_split_labels = argw["fix_split_labels"])
    return data_, data_.__get_info__()

