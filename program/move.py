import os

for folder in os.listdir('captures'):
    for fileName in os.listdir('captures/{}'.format(folder)):
        print(fileName)
        os.rename('captures/{}/{}'.format(folder,fileName), '../../../tf_files/ASL_test/{}'.format(fileName))
