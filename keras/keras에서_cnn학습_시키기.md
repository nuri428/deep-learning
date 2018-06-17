#Keras에서 CNN학습 시키기 
keras를 이용해서 CNN(Convolution Neural Network)학습하기. 

1. 이 내용에서는 다음 [링크](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)에 있는 VGG16이라는 모델을 이용합니다. 
2. 예제 코드는 이 [링크](./vgg-16-learn.py)을 통해 확인 할 수 있습니다. 
3. 예제 데이타는 ./dataroot/class/*.jpg 와 같은 형식이며 이 프로그램의 인자는 dataroot를 입력 받습니다. 
4. 예제 데이타의 이미지 파일(jpg)은 데이타를 로딩하는 과정에서 자동으로 vgg16모델의 사이즈로 리사이즈 됩니다. 
5. 예제 코드내에서 수정해야 할 사항은 nbClasses 변수의 값이며, 입력되는 테스트 데이터의 클래스 +1로 설정 하셔야 합니다. 


1. 이미지 데이터를 keras에서 사용하는 데이터 형태로 로딩 하기 

def load_data (data_path):  

    columns = ['data','label']
    df = pandas.DataFrame(columns=columns)
    for root, subFolder, files in os.walk(data_path):
        for item in files:
            if item.endswith(".jpg"):
                fileNamePath = str(os.path.join(root,item))
                im = cv2.imread(fileNamePath)
                im = cv2.resize(im, (input_width,input_height))
                imgArray = np.asarray(im)
                classStr = int("".join(fileNamePath.split("/")[-2:-1]))
                df.loc[len(df)] = [imgArray, classStr]
                if len(df) % 10000 == 0 :
                    print '%d load '%len(df)
    print 'load %d data '%len(df)
    data = np.array(df['data'].tolist())
    label = np.array(df['label'].tolist())
    data = np.transpose(data, (0,3,1,2)) # 중요
    return data, label
    
CNN을 위한 keras 데이타 입력 형식은 image(numpy.array), class(int) 형식의 쌍으로 구성되어 있으며, 괄호는 데이타 타입을 의미합니다. 

위의 코드에서 for 문으로 시작하는 코드는 사용자로부터 데이타가 있는 경로를 입력 받아 해당 경로내의 모든 jpg파일을 읽어 들이고, jpg파일을 input_width, input_height로 크기를 수정한후, 클래스 정보를 추출하여 pandas.DataFrame객체에 각각의 컬럼 data, label에 저장합니다.

이렇게 저장된 데이터를 이미지 데이타와 클래스를 분리하여 반환하는데, 그 사이에 data = np.transpose(data,(0,3,1,2))라는 구문은 keras에서 이미지를 읽어 들이기 형식에 맞게 데이터를 변형하는 단계입니다. 

 

 
