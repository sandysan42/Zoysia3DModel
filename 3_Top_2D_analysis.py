import imutils
import pathlib
import csv
import cv2

path = pathlib.Path('MASK_FOLDER')
csv_path ='result.csv'
output_path = './'
ratio = 0.08305 #mm/pixel
line_size = 5
drawcontour = True

def CreateContour(binary):
    cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
    if len(cntsSorted) != 0:
        return cntsSorted[-1]

def main():
    folder_list = list(path.glob('*'))
    folder_list.sort()
    for date_fold in folder_list:
        input_list = list(date_fold.glob('*.JPG'))
        input_list.sort()
        date = str(date_fold.relative_to(path))
        opd_path = pathlib.Path(output_path,date)
        opd_path.mkdir(exist_ok=True)
        for im in input_list:
            img = cv2.imread(str(im))
            img_name = str(im.relative_to(date_fold))
            img_name = img_name[0:-4]

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            (T,binary) = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

            c = CreateContour(binary)
            M = cv2.moments(c)
            A = cv2.contourArea(c)

            if drawcontour == True:
                cv2.drawContours(img, [c], -1, (0, 255, 255), line_size)
            cv2.imwrite(str(pathlib.Path(output_path,date,img_name+'_contour.png')), img)

            A = A * (ratio**2)/100

            write = [img_name,A] 
            with open(csv_path, 'a', newline='\n') as file_object:
                file_write = csv.writer(file_object)
                file_write.writerow(write)
                file_object.close() 

if __name__ == '__main__':
    main()