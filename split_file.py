from  constants import  *
if __name__ == '__main__':

    fw = open(less_train_feature,'w+')
    i = 0
    with open(train_feature_filna_path,'r') as f:
        for line in f.readlines()[0:3000]:
            try:

                fw.write(line)
                i += 1
                if i % 100 == 0:
                    print(i)
            except Exception as e:
                print(e)



        fw.close()
        f.close()