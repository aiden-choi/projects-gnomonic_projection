# image projected from fisheye to equirectangular image.
import cv2
import numpy as np
import time

class Image_proj:
    def __init__(self, sample_path, sph_file):
        self.sample_path = sample_path
        self.spherical_path = sample_path+sph_file
        self.sph_img = []

        self.sph_w = 0      # x (longitude)
        self.sph_h = 0      # y (latitude)
        self.sph_center = [0,0] # [x,y]
        self.sph_latitude_range = [-90, 90]       # y
        self.sph_longitude_range = [-180, 180]    # x

        self.fov = [60,60]  # suppose fov 60 deg. [x,y]
        self.output_w = 320
        self.output_h = 320

    def load_sph_image(self):
        self.sph_img = cv2.imread(self.spherical_path)
        # self.sph_img = cv2.cvtColor(self.sph_img, cv2.COLOR_BGR2GRAY)

        # print(self.sph_img.shape)
        img_size = self.sph_img.shape
        self.sph_w = img_size[1]
        self.sph_h = img_size[0]
        self.sph_center = [int(self.sph_w/2), int(self.sph_h/2)]

        # cv2.imshow('sph_img',self.sph_img)
        # cv2.waitKey(0)

    def gnomonic_proj(self, latitude, longitude):
        # matrix version of gnomonic projection
        # http://mathworld.wolfram.com/GnomonicProjection.html
        # https://en.wikipedia.org/wiki/Gnomonic_projection

        latit1 = -1 * latitude / 180 * np.pi    # multiply -1 to consider array index of python [0,0]-> top left.
        long0 = longitude / 180 * np.pi

        output_img = np.zeros([self.output_h, self.output_w, 3])
        img_size = output_img.shape

        x = np.array(range(img_size[1]))
        y = np.array(range(img_size[0]))
        r_x = 2 * (x + 0.5) / img_size[1] - 1.0 # 0.5 for preventing zero divide
        r_y = 2 * (y + 0.5) / img_size[0] - 1.0 # 0.5 for preventing zero divide

        x_fov_ratio = self.fov[0] / 2 / 180
        y_fov_ratio = self.fov[1] / 2 / 180
        r_x = r_x * x_fov_ratio * np.pi
        r_y = r_y * y_fov_ratio * np.pi

        rho_x = np.tile(r_x * r_x, (len(r_y), 1))
        rho_y = np.transpose(np.tile(r_y * r_y, (len(r_x), 1)))
        rho = np.sqrt(rho_x+rho_y)
        c = np.arctan(rho)       
        
        sin_latit1 = np.sin(latit1)
        cos_latit1 = np.cos(latit1)
        cos_c = np.cos(c) # bottleneck
        sin_c = np.sin(c) # bottleneck

        X1 = cos_c*sin_latit1 + r_y[:,None]*sin_c*cos_latit1/rho
        cur_latit = np.arcsin(X1) # bottleneck

        X2 = r_x*sin_c
        X3 = (rho*cos_latit1*cos_c - r_y[:,None]*sin_latit1*sin_c)
        cur_long = long0 + np.arctan2(X2,X3) # bottleneck
        
        sph_x = cur_long / (2 * np.pi) + 0.5    # 2*pi -> -180 ~ 180. and +0.5 for shift
        sph_y = cur_latit / np.pi + 0.5         # pi -> -90 ~ 90.   and +0.5 for shift

        sph_x = np.array(sph_x*self.sph_w,dtype=np.int)
        sph_y = np.array(sph_y*self.sph_h,dtype=np.int)

        mat_append = (sph_x >= self.sph_w) * self.sph_w
        sph_x = sph_x - mat_append
        mat_append = (sph_x < 0) * self.sph_w
        sph_x = sph_x + mat_append

        mat_append = (sph_y >= self.sph_h) * self.sph_h
        sph_y = sph_y - mat_append
        mat_append = (sph_y < 0) * self.sph_h
        sph_y = sph_y + mat_append

        y_ind = np.transpose(np.tile(y, (len(x),1)))
        x_ind = np.tile(x, (len(y),1))

        output_img[y_ind,x_ind,:] = self.sph_img[sph_y,sph_x,:] # bottleneck
        output_img= np.uint8(output_img)


        return output_img

if __name__ == '__main__':

    # load
    sample_path = "../samples/"
    sph_file = "sample.jpg"

    # parameters


    img_projection = Image_proj(sample_path,sph_file)
    img_projection.load_sph_image()

    # out_1 = img_projection.gnomonic_proj(0, 90)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for latit in range(-90,90,10):
        for long in range(-180,180,10):
            t1 = time.time()
            out_2 = img_projection.gnomonic_proj(latit,long)
            t2 = time.time()
            cv2.putText(out_2, 'latit: ' + str(latit) + ' long: ' + str(long) + ' time: ' + str((t2 - t1) * 1000),
                        (10, 30), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(out_2, 'latit: ' + str(latit) + ' long: ' + str(long),
            #             (10, 30), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('a',out_2)
            cv2.waitKey(0)


