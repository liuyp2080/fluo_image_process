#fluo_img_preprocessing

import numpy as np
import skimage
import streamlit as st
from PIL import Image
from skimage import exposure
import io
from skimage.color import gray2rgb, rgb2gray, rgb2hed
from skimage.color import rgb2hed, hed2rgb,hed2rgb
from skimage.exposure import match_histograms
from skimage.color import (separate_stains, combine_stains,
                           hdx_from_rgb, rgb_from_hdx)
# import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.restoration import (
    calibrate_denoiser,
    denoise_wavelet,
    denoise_tv_chambolle,
    denoise_nl_means,
    denoise_bilateral,
    estimate_sigma,
)
from skimage.filters import rank
from skimage.metrics import peak_signal_noise_ratio

#------------------------函数-----------------------------------
def download_image(image, file_name="img_channel", file_format="tiff"):
    """
    Download the given image as a file.

    Parameters
    ----------
    image : array-like
        The image to be downloaded.
    file_name : str, optional
        The name of the file to be downloaded. Defaults to "img_channel".
    file_format : str, optional
        The format of the file to be downloaded. Defaults to "tiff".
    """
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    buffer = io.BytesIO()
    image_pil.save(buffer, format=file_format)
    data = buffer.getvalue()
    return st.download_button(
        label="Download Image",
        data=data,
        file_name=f"{file_name}.{file_format}",
        key=file_name
    )
def equalization(img_channel):
    img_channel = rgb2gray(img_channel)
    img_channel = exposure.equalize_adapthist(img_channel, clip_limit=0.03)
    img_channel = gray2rgb(img_channel)
    return img_channel
#---------------------------------------UI-----------------
set_page_config = st.set_page_config(
    page_title="荧光图片处理",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🧙‍♂️荧光图片处理APP") 

with st.sidebar.container(border=True):
    st.header('APP介绍')
    '''
    这个APP可以用来处理多通道的荧光图片，功能包括：
    1. 图片降噪，减弱因为各种原因导致的图片上的噪点，改善清晰度；
    2. 图片曝光度调节，改善对比度；
    3. 通道颜色交换，红绿蓝三个通道之间的颜色进行互换；
    4. 免疫组化图片转荧光图片。
    '''
with st.sidebar.container(border=True):
    st.header('关注作者')
    st.write('''
            📰微信公众号：医研趣与美；\n
            📰CSDN账号：医学预测模型的开发与应用研究; \n
            🌐医学APP矩阵：app.clinicalmodelmatrix.com（需账号）
                 ''')
    
    
Tab1, Tab2, Tab3, Tab4 = st.tabs(["🖼️降噪和曝光度调整", "🎨通道颜色调整", "🔀免疫组化转荧光","👥免疫组化颜色匹配"])

with Tab1:

    #first, use median filter, to denoise the image
    # show the image
    st.header("🖼️图片降噪")
    '''
    提高图片的清晰度，是图片处理的第一步。提供了多种不同的方法来降噪，可以从中选择最佳的方法。
    '''
    uploaded_file = st.file_uploader("选择图片...", type=["jpg", "png", "tif", "tiff","bmp"])
    if uploaded_file is not None:
        img = skimage.io.imread(uploaded_file)
    else:
        img = skimage.io.imread('S-p-mTOR_filter_median_1.2_contrast.tif')
   
    col1, col2 = st.columns(2)
    with col1:    
        with st.container(border=True):
            channel = st.selectbox("Select Channel", options=['Channel All',"Channel Red", "Channel Green", "Channel Blue"])
            if img.ndim == 3:
                img=img
            else:
                img= gray2rgb(img)
            null = np.zeros_like(img[:, :, 0])
            if channel == 'Channel Red':
                img = np.dstack((img[:, :, 0], null, null))
            elif channel == 'Channel Green':
                img = np.dstack((null, img[:, :, 1], null))
            elif channel == 'Channel Blue':
                img = np.dstack((null, null, img[:, :, 2]))
            else:
                img=img
            st.image(img)
    with col2:
        with st.container(border=True):
            denoise_selector = st.selectbox("选择降噪方法", 
                                            options=['None',"Bilateral Filter", "Wavelet Filter", "TV Filter", "NL Means Filter"])
            img = img_as_float(img)
            sigma_est = estimate_sigma(img, average_sigmas=True)
            if denoise_selector == "Bilateral Filter":
                img_denoised=denoise_bilateral(img, sigma_color=0.05,sigma_spatial=15,channel_axis=-1)
                st.image(img_denoised)
            elif denoise_selector == "Wavelet Filter":
                img_denoised=denoise_wavelet(img, convert2ycbcr=True,channel_axis=-1, rescale_sigma=True)
                st.image(img_denoised,clamp=True)
            elif denoise_selector == "TV Filter":
                # weight= st.slider("Weight", min_value=0.1, max_value=1.0, value=0.1, step=0.1)
                img_denoised=denoise_tv_chambolle(img, weight=0.1, channel_axis=-1)
                st.image(img_denoised,clamp=True)
            elif denoise_selector == "NL Means Filter":
                img_denoised=denoise_nl_means(img, h=0.8*sigma_est, sigma=sigma_est, fast_mode=True, channel_axis=-1)
                st.image(img_denoised)
            else:
                img_denoised=img
                st.image(img_denoised,clamp=True)
    st.divider()
    st.header("🌄曝光度调整")
    '''
    提高图片的对比度，是图片处理的第二步。
    '''
    col3,col4,col5 = st.columns(3)
    with col3:
        with st.container():
            null = np.zeros_like(img[:, :, 0])
            st.write('Channel Red')
            s_red1,s_red2 = st.slider("Channel Red", min_value=0.0, max_value=100.0, value=(2.0, 98.0), step=0.1)
            img_channel_red = np.dstack((img_denoised[:, :, 0], null, null))
            p_red1, p_red2 = np.percentile(img_channel_red, (s_red1, s_red2))
            img_channel_red= exposure.rescale_intensity(img_channel_red, in_range=(p_red1, p_red2))
            # Equalization
            check_channel_red=st.checkbox('Equalization', value=False, key='equalization_red')
            if check_channel_red:
                img_channel_red = equalization(img_channel_red)
                img_channel_red = np.dstack((img_channel_red[:, :, 0], null, null))
            st.image(img_channel_red,clamp= True)
            download_image(img_channel_red, file_name="img_channel_red", file_format="tiff")


    with col4:  
        with st.container():
            st.write('Channel Green')
            s_green1,s_green2 = st.slider("Channel Green", min_value=0.0, max_value=100.0, value=(2.0, 98.0), step=0.1)
            img_channel_green = np.dstack((null, img_denoised[:, :, 1], null))
            p_green1, p_green2 = np.percentile(img_channel_green, (s_green1, s_green2))
            img_channel_green= exposure.rescale_intensity(img_channel_green, in_range=(p_green1, p_green2))
            # Equalization
            check_channel_green=st.checkbox('Equalization', value=False, key='equalization_green')
            if check_channel_green:
                img_channel_green = equalization(img_channel_green)
                img_channel_green = np.dstack((null, img_channel_green[:, :, 1], null))
            st.image(img_channel_green)
            download_image(img_channel_green, file_name="img_channel_green", file_format="tiff")
 
    with col5:  
        with st.container():
            st.write('Channel Blue')
            s_blue1,s_blue2 = st.slider("Channel Blue", min_value=0.0, max_value=100.0, value=(2.0, 98.0), step=0.1)
            img_channel_blue = np.dstack((null, null, img_denoised[:, :, 2]))
            p_blue1, p_blue2 = np.percentile(img_channel_blue, (s_blue1, s_blue2))
            img_channel_blue= exposure.rescale_intensity(img_channel_blue, in_range=(p_blue1, p_blue2))
            # Equalization
            check_channel_blue=st.checkbox('Equalization', value=False, key='equalization_blue')
            if check_channel_blue:
                img_channel_blue = equalization(img_channel_blue)
                img_channel_blue = np.dstack((null, null, img_channel_blue[:, :, 2]))
            st.image(img_channel_blue)
            download_image(img_channel_blue, file_name="img_channel_blue", file_format="tiff")
  
    #combing the three channels into one image
    img_preview = np.dstack((img_channel_red[:, :, 0], img_channel_green[:, :, 1], img_channel_blue[:, :, 2]))
    #save the image—tif
    st.divider()

    download_image(img_preview, file_name="img_preview", file_format="tiff")
    # #download the image
    with st.container(height=600):
        st.header("Output Image")
        st.image(img_preview,channels="RGB",output_format="tiff")
    st.warning('提示：图片可以使用下载按键进行下载，也可以在图片上点击右键保存，后者分辨率更佳。')
with Tab2:
    #channel exchange ,such as red change to green
    st.header("🎨通道颜色交换")
    uploaded_file2 = st.file_uploader("选择需要交换颜色的图片...", type=["jpg", "png", "tif", "tiff",'bmp'])
    if uploaded_file2 is not None:
        img_color = skimage.io.imread(uploaded_file2)
    else:
        img_color = skimage.io.imread('S-p-mTOR_filter_median_1.2_contrast.tif')
    null_color = np.zeros_like(img_color[:, :, 0])
    col6, col7 , col8= st.columns(3)
    with col6:
        with st.container():
            n0=st.select_slider("Channel 0", options=['Channel red', 'Channel green', 'Channel blue'], value='Channel red')
            if n0=='Channel red':
                img_channel_0= np.dstack((img_color[:, :, 0], null_color, null_color))
                st.image(img_channel_0)
            elif n0=='Channel green':
                img_channel_0= np.dstack((null_color, img_color[:, :, 0], null_color))
                st.image(img_channel_0)
            else:
                img_channel_0= np.dstack((null_color, null_color, img_color[:, :, 0]))
                st.image(img_channel_0)
    with col7:
        with st.container():
            n1=st.select_slider("Channel 1", options=['Channel red', 'Channel green', 'Channel blue'], value='Channel green')
            if n1=='Channel red':
                img_channel_1= np.dstack((img_color[:, :, 1], null_color, null_color))
                st.image(img_channel_1)
            elif n1=='Channel green':
                img_channel_1= np.dstack((null_color, img_color[:, :, 1], null_color))
                st.image(img_channel_1)
            else:
                img_channel_1= np.dstack((null_color, null_color, img_color[:, :, 1]))
                st.image(img_channel_1)
    with col8:
        with st.container():
            n2=st.select_slider("Channel 2", options=['Channel red', 'Channel green', 'Channel blue'], value='Channel blue')
            if n2=='Channel red':    
                img_channel_2= np.dstack((img_color[:, :, 2], null_color, null_color))
                st.image(img_channel_2)
            elif n2=='Channel green':
                img_channel_2= np.dstack((null_color, img_color[:, :, 2], null_color))                
                st.image(img_channel_2)
            else:                
                img_channel_2= np.dstack((null_color, null_color, img_color[:, :, 2]))
                st.image(img_channel_2)

    st.divider()
    with st.container(height=600):
        #n0等对应转换后的通道，值0，1，2，对应转换前的通道
        if n0=='Channel red' and n1=='Channel green' and n2=='Channel blue':
            img_combine=np.dstack((img_channel_0[:, :, 0], img_channel_1[:, :, 1], img_channel_2[:, :, 2]))
            st.image(img_combine)
        elif n0=='Channel green' and n1=='Channel red' and n2=='Channel blue':
            img_combine=np.dstack((img_channel_1[:, :, 0], img_channel_0[:, :, 1], img_channel_2[:, :, 2]))
            st.image(img_combine)
        elif n0=='Channel blue' and n1=='Channel green' and n2=='Channel red':
            img_combine=np.dstack((img_channel_2[:, :, 0], img_channel_1[:, :, 1], img_channel_0[:, :, 2]))
            st.image(img_combine)
        elif n0=='Channel red' and n1=='Channel blue' and n2=='Channel green':
            img_combine=np.dstack((img_channel_0[:, :, 0], img_channel_2[:, :, 1], img_channel_1[:, :, 2]))
            st.image(img_combine)
        elif n0=='Channel blue' and n1=='Channel red' and n2=='Channel green':#
            img_combine=np.dstack((img_channel_1[:, :, 0], img_channel_2[:, :, 1], img_channel_0[:, :, 2]))
            st.image(img_combine)
        elif n0=='Channel green' and n1=='Channel blue' and n2=='Channel red':#
            img_combine=np.dstack((img_channel_2[:, :, 0], img_channel_0[:, :, 1], img_channel_1[:, :, 2]))
            st.image(img_combine)
        download_image(img_combine, file_name="img_combine", file_format="tiff")
                
    st.warning('提示：图片可以使用下载按键进行下载，也可以在图片上点击右键保存，后者分辨率更佳。')
#------------------------免疫组化转荧光----------------------------------------------------------------------------
with Tab3:
    st.header("🔀免疫组化转荧光")
    '''
    该模块可以将免疫组化图片转化为荧光图片，喜欢荧光图片的可以尝试。
    '''
    uploaded_file3 = st.file_uploader("上传待转换的图", type=["jpg", "png", "tif", "tiff",'bmp'])
    if uploaded_file3 is not None:
        image_orginal = skimage.io.imread(uploaded_file3)
    else:
        image_orginal = skimage.io.imread('5s-2h-20um-2.bmp')
            
    col9, col10, col11 = st.columns(3)
    col12,col13,col14 = st.columns(3)
    with col9:
        st.write('原始图像')
        st.image(image_orginal)
    ihc_hed = rgb2hed(image_orginal)
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    with col12:
        st.write('苏木素通道')
        s_h1,s_h2 = st.slider("Channel H", min_value=0.0, max_value=100.0, value=(2.0, 99.8), step=0.1)
        p_h1, p_h2 = np.percentile(ihc_hed[:, :, 0], (s_h1, s_h2))
        h = exposure.rescale_intensity(
            ihc_hed[:, :, 0],
            out_range=(0, 1),
            in_range=(p_h1, p_h2)
                )
        st.image(ihc_h)
    with col13:
        st.write('伊红通道')
        s_e1,s_e2 = st.slider("Channel E", min_value=0.0, max_value=100.0, value=(2.0, 99.8), step=0.1)
        p_e1, p_e2 = np.percentile(ihc_hed[:, :, 1], (s_e1, s_e2))
        e = exposure.rescale_intensity(
            ihc_hed[:, :, 1],
            out_range=(0, 1),
            in_range=(p_e1, p_e2),
                )
        st.image(ihc_e)
    with col14:
        st.write('DAB通道')
        s_d1,s_d2 = st.slider("Channel D", min_value=0.0, max_value=100.0, value=(2.0, 99.8), step=0.1)
        p_d1, p_d2 = np.percentile(ihc_hed[:, :, 2], (s_d1, s_d2))
        d = exposure.rescale_intensity(
            ihc_hed[:, :, 2],
            out_range=(0, 1),
            in_range=(p_d1, p_d2),
        )
        st.image(ihc_d)
    with col10:
        fluo_mode=st.radio('荧光搭配模式',options=['DAB+苏木素+伊红','苏木素+伊红','苏木素+DAB'],
                        captions=['DAB(绿)+苏木素(蓝)+伊红(红)','苏木素(蓝)+伊红(红)','苏木素(蓝)+DAB(绿)'])
            
        st.info('提示：更多颜色搭配可以使用“荧光颜色转换”模块')
    with col11:
        st.write('荧光图片')
    
        if fluo_mode=='DAB+苏木素+伊红':
            zdh = np.dstack((e, d, h))
            st.image(zdh)
        elif fluo_mode=='苏木素+伊红':
            zdh = np.dstack((e, null, h))
            st.image(zdh)
        elif fluo_mode=='苏木素+DAB':
            zdh = np.dstack((null, d, h))
            st.image(zdh)
            
with Tab4:
    #color histogram matching
    st.header("🔀颜色直方图匹配")
    '''
    可用于统一免疫组化图片之间细微的色差问题，效果有待于进一步考证。
    '''
    col_c,col_d,col_e = st.columns(3)
    col_f,col_g,col_h = st.columns(3)
    with col_c:
        uploaded_file4 = st.file_uploader("选择待匹配图像...", type=["jpg", "png", "tif", "tiff","bmp"])
        if uploaded_file4 is not None:
            image = skimage.io.imread(uploaded_file4)
        else:
            image = skimage.io.imread('5s-2h-20um-1.bmp')
    with col_d:
        uploaded_file5 = st.file_uploader("选择参考图像...", type=["jpg", "png", "tif", "tiff","bmp"])
        if uploaded_file5 is not None:    
            reference = skimage.io.imread(uploaded_file5)
        else:
            reference = skimage.io.imread('5s-2h-20um-2.bmp')
    with col_f:
        st.write('待匹配图像')
        # image_hdx=separate_stains(image,hdx_from_rgb,channel_axis=-1)
        st.image(image,clamp=True)
    with col_g:
        st.write('参考图像')
        # reference_hdx=separate_stains(reference,hdx_from_rgb,channel_axis=-1)
        st.image(reference,clamp=True)
    with col_h:
        st.write('匹配结果')
        matched= match_histograms(image, reference, channel_axis=-1)
        # matched_rgb=combine_stains(matched_hed,rgb_from_hdx,channel_axis=-1)
        st.image(matched,clamp=True)