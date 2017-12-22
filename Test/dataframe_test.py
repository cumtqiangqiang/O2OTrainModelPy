import  pandas as pd
import  numpy as np

if __name__ == '__main__':
    d = {'col1':['a','b','c','a','b'],'col2':['a','b','c','a','d']
         ,'col3':np.arange(5)
         }
    df = pd.DataFrame(data=d)
    print(df)
    # grop = df.groupby('col1').mean()
    # print(grop)
    print('---------------------------')

    for (k1,k2),group in df.groupby(['col1','col2']):
        print(k1,k2)
        print(group)





