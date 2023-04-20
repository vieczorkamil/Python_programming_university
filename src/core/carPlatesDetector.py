from typing import Union, Tuple
import numpy as np
import cv2
import easyocr


class CarPlatesDetector:
    def __init__(self, modelWeightsPath: str, modelConfigPath: str) -> None:
        self.minimumScore = 0.5
        self.threshold = 0.3
        self.ocrCustomConfig = r'--oem 3 --psm 6'
        
        self.results = None
        self.resetImage()
        self._loadModl(weightsPath=modelWeightsPath, configPath=modelConfigPath)


    def test(self) -> str:
        return "OKI it works with fastapi"


    def setInputImage(self, inputImg: cv2.Mat) -> None:
        self.inputImg = inputImg


    def loadImage(self, inputImagePath: str) -> None:
        self.inputImg = cv2.imread(inputImagePath)
        self.inputImg = cv2.cvtColor(self.inputImg, cv2.COLOR_BGR2RGB)


    def resetImage(self) -> None:
        self.inputImg = None


    def process(self) -> Tuple[str, Union[np.ndarray, any], Union[np.ndarray, any], Union[str, any]]:
        img = None
        imgPlate = None
        plateText = None
        retValue = "Not found"
        blob = self._preProcessing()
        if self._search(networkInput=blob):
            for i in self.results.flatten():
                xMin, yMin = self.boundingBoxes[i][0], self.boundingBoxes[i][1]
                boxWidth, boxHeight = self.boundingBoxes[i][2], self.boundingBoxes[i][3]
                img = self._getResultImage(x=xMin, y=yMin, w=boxWidth, h=boxHeight)
                imgPlate = self._getResultPlateImage(x=xMin, y=yMin, w=boxWidth, h=boxHeight)
                plateText = self._getResultPlateText(imgPlate)
                self.resetImage()
                retValue = "Found"
            
        if retValue == "Found":
            return [retValue, img, imgPlate, plateText]
        else:
            return [retValue, None, None, None]


    def _search(self, networkInput: np.ndarray) -> bool:
        layersAllNames = self.network.getLayerNames()
        layersOutputNames = [layersAllNames[i-1] for i in self.network.getUnconnectedOutLayers()]
        self.network.setInput(networkInput)
        self.networkOutput = self.network.forward(layersOutputNames)
        self.boundingBoxes = []
        self.confidences = []
        self.classNumbers = []
        h, w = self.inputImg.shape[:2]
        retValue = False

        if len(self.networkOutput):
            for result in self.networkOutput:
                for detection in result:
                    scores = detection[5:]
                    classCurrent = np.argmax(scores)
                    confidenceCurrent = scores[classCurrent]
                    if confidenceCurrent > self.minimumScore:
                        boxCurrent = detection[0:4] * np.array([w, h, w, h])
                        xCenter, yCenter, boxWidth, boxHeight = boxCurrent.astype('int')
                        xMin = int(xCenter-(boxWidth/2))
                        yMin = int(yCenter-(boxHeight/2))
                        self.boundingBoxes.append([xMin, yMin, int(boxWidth), int(boxHeight)])
                        self.confidences.append(float(confidenceCurrent))
                        self.classNumbers.append(classCurrent) 
                        retValue = True

        if retValue:
            self.results = cv2.dnn.NMSBoxes(self.boundingBoxes, self.confidences, self.minimumScore, self.threshold)
        return retValue


    def _preProcessing(self) -> np.ndarray:
        if self.inputImg is not None:
            return cv2.dnn.blobFromImage(self.inputImg, scalefactor=1/255.0, size=(416,416), swapRB=True, crop=False)


    def _getResultImage(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        resultImg = self.inputImg.copy()
        cv2.rectangle(img=resultImg, pt1=(x, y), pt2=(x+w, y+h), color=(51, 255, 51), thickness=5)
        return resultImg


    def _getResultPlateImage(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        imgCropped = self.inputImg[y: y+h, x:x+w]
        return imgCropped


    def _getResultPlateText(self, imgInput) -> str:
        imgInput = cv2.cvtColor(imgInput, cv2.COLOR_RGB2GRAY)

        reader = easyocr.Reader(['en'])
        plateNumber = reader.readtext(imgInput, detail = 0)[0]

        weirdChar = [" ","'","[", "]","{", "}",",",".","/","?","~","`","\\","\"","(",")","!","|","-","_","@"]
        for i in weirdChar:
            plateNumber = plateNumber.replace(i, "")

        return plateNumber


    def _loadModl(self, weightsPath: str, configPath: str) -> None:
        self.network = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# pure API test
# if __name__ == "__main__":
#     cfg = "./models/darknet-yolov3.cfg"
#     weights = "./models/lapi.weights"
#     detector = CarPlatesDetector(weights, cfg)
#     detector.loadImage("../photos/car2.jpg")
    
#     result = detector.process()


#     if result:
#         [response, img1, img2, txt] = result
#         response = {"Is result found": "True",
#                     "Plate number": txt}
        
#         fig, (ax1, ax2) = plt.subplots(2)
#         fig.suptitle(f'Plate number detection result: {txt}')
#         ax1.imshow(img1)
#         ax2.imshow(img2)
#         plt.show()
#     else:
#         response = {"Is result found": "False"}    

#     jsonResponse = json.dumps(response, indent = 4) 
#     print(jsonResponse)