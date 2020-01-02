import numpy as np
from photutils import aperture_photometry
from photutils import CircularAperture
from astropy.io import fits
import sys
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy.ma as ma
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
import os
from photutils import centroid_com, centroid_1dg, centroid_2dg
from exoplanets_dot_org import *
import pandas as pd
import socket


def main(plnm, channel, aornum):

    if __name__ == '__main__':
        basepath='/home/bkilpatr/mapping_files/'
        if socket.gethostname()=='Luke': basepath='/Users/Brian/Desktop/Tucker_Group/Spitzer/mapping_files/'
        if socket.gethostname()=='Pollux.local': basepath='/Users/bmkil/Mapping/'

        if plnm == 'HD209': plnm2='HD 209458 b'
        if plnm == 'HD189': plnm2='HD 189733 b'
        if plnm== 'W16': 
            plnm2='WASP-16 b'
            basepath='/Users/Brian/Desktop/Tucker_Group/Spitzer/mapping_files/'
        if plnm== 'W15': 
            plnm2='WASP-15 b'
            basepath='/Users/Brian/Desktop/Tucker_Group/Spitzer/mapping_files/'

        if plnm== 'TRAPPIST-1': 
            plnm2='TRAPPIST-1 b'
            basepath='/Users/Brian/Desktop/Tucker_Group/t_1/'

        if plnm== 'HAT_P_41': 
            plnm2='HAT-P-41 b'
            basepath='/Users/Brian/Desktop/Tucker_Group/Spitzer/HAT_P_41/'


        basepath='/Users/bmkil/Documents/Spitzer/W121/'
        plnm2='WASP-121 b'


        chnum=channel#1
      
        extract='on'
        gaussiancent='off' 
        
        
        radii=np.arange(10)/10.+1.8
        var_rad=np.arange(10)/20.+0.7
        var_add=np.arange(10)/10-0.6

        if aornum[0]=='none':aornum=aor_from_list(plnm, chnum, basepath)
        if aornum[0]=='find': 
            print ('in')
            aorlist=glob.glob(basepath+'/r*')
            for i in range(len(aorlist)):
                item=aorlist[i]
                aorlist[i]=item[-8:]
                aornum=aorlist

        print(aornum)

        if extract=='on':  
            extraction(aornum, chnum, plnm, plnm2, radii, var_rad, var_add, gaussiancent, basepath)


def extraction(aornum, chnum, plnm, plnm2, radii, var_rad, var_add, gaussiancent, basepath):

    for aor in aornum:
        aor=str(aor)

        ch=str(os.listdir(basepath+'r'+aor+ '/')[0])
        chnum=ch[-1:]

        # if socket.gethostname()=='Pollux.local':
        #     paths = glob.glob(basepath+ 'image_files/'+'r'+aor+ '/ch*')
        paths = glob.glob(basepath+'r'+aor+ '/ch*')
        fpathout=basepath+'outputs/'+plnm+'/Ch_'+str(chnum)+'/'+aor+'/'
        directory = os.path.dirname(fpathout)
        if not os.path.exists(directory):
            os.makedirs(directory)
 

        fpath=paths[0]+'/bcd/'


        filenames=glob.glob(fpath+ '*bcd.fits')
        nframes=len(filenames)

     
        hold_pos=np.zeros(shape=(nframes*64, 2))
        central_pix=np.zeros(shape=(3,3, nframes*64))
        #all_rad=np.concatenate((radii, var_rad))
        lightcurve=np.zeros(shape=(nframes*64, len(radii)+len(var_rad)+len(var_add)))
        time=np.zeros(nframes*64)
        beta_np=np.zeros(nframes*64)

        orbparams=get_orb_pars(plnm2, basepath)
               
        for i in range(0,nframes):
            if i % 10==0: 
                os.system('clear')
                print(aor, i,' of ',str(nframes))

            hdulist = fits.open(filenames[i])
            channel=str(hdulist[0].header['CHNLNUM'])
            gain=str(hdulist[0].header['GAIN'])
            fluxconv=str(hdulist[0].header['FLUXCONV'])
            
            cube=hdulist[0].data
            exptime=hdulist[0].header['EXPTIME']
            framtime=hdulist[0].header['FRAMTIME']
            mmbjd=hdulist[0].header['BMJD_OBS']
          
            for j in range(0,64):    
                scidata=cube[j,:,:]
                bkgd= backgr(scidata)
                data=scidata-bkgd  
                data=ma.masked_invalid(data)
                
                bnp1=np.sum(data)**2
                bnp2=np.sum(np.multiply(data,data))
                bnp=bnp1/bnp2
        
                xc,yc=centroid(data, gaussiancent)
                position=[xc,yc]
                beta_np[64*i+j]=bnp
                hold_pos[64*i+j, :]=position  
                vrad1=var_rad*np.sqrt(bnp)
                vrad2=var_add+np.sqrt(bnp)
                vrad=np.concatenate((vrad1, vrad2))
                all_rad=np.concatenate((radii, vrad))

         

                if np.sqrt(bnp)<1.: lightcurve[64*i+j, :]=np.nan
                else:
                    apertures = [CircularAperture(position, r=r) for r in all_rad]

                    phot_table = aperture_photometry(scidata, apertures)

                    for q in range (0,len(all_rad)):
                        if q ==0: phot=np.zeros(len(all_rad))
                        phot[q]=phot_table.columns[q+3]  
                    
                    lightcurve[64*i+j, :]=phot
                time[64*i+j]=mmbjd+framtime*j/86400.0
                if (i==0) & (j==0):  max_pix_x, max_pix_y=int(np.floor(xc)), int(np.floor(yc))


                central_pix[:,:,64*i+j]=data[max_pix_x-1:max_pix_x+2, max_pix_y-1:max_pix_y+2]/np.sum(data[max_pix_x-1:max_pix_x+2, max_pix_y-1:max_pix_y+2])
                #print (central_pix[:,:, 64*i+j],np.sum(central_pix[:,:, 64*i+j]))
          
            hdulist.close()
        

        #pmap=pmap_corr(hold_pos, channel)
        order=np.argsort(np.squeeze(time))            
        time=time[order]
        lightcurve=lightcurve[order, :]
        hold_pos=hold_pos[order, :]
        beta_np=beta_np[order]
        central_pix=central_pix[:, :, order]
 
        np.savez(fpathout+'extraction',  ch=channel, time=time, lc=lightcurve, cp=central_pix,  op=orbparams, exptime=exptime, framtime=framtime, beta_np=beta_np, hold_pos=hold_pos, all_rad=all_rad, gain=gain, fcon=fluxconv)
        t=time
        npix=beta_np

        plt.figure()
        plt.subplot(411)
        plt.title(plnm+' Ch: '+str(chnum)+'\n'+aor)
        plt.scatter(t, hold_pos[:,0], s=1)
        plt.ylim(np.nanmedian(hold_pos[:,0])-0.5, np.nanmedian(hold_pos[:,0])+0.5)
        plt.ylabel('X position')
        plt.xticks([])
        plt.subplot(412)
        plt.scatter(t, hold_pos[:,1], s=1)
        plt.ylim(np.nanmedian(hold_pos[:,1])-0.5, np.nanmedian(hold_pos[:,1])+0.5)
        plt.ylabel('Y position')
        plt.xticks([])
        plt.subplot(413)
        plt.scatter(t, np.sqrt(npix), s=1)
        plt.ylim(2, 3)
        plt.ylabel('Sqrt Noise Pixel')
        plt.xlabel('Time')
        plt.subplot(414)
        plt.scatter(t, lightcurve[:,5]/np.nanmedian(lightcurve[:,5]), s=1)
        plt.ylim(0.97, 1.03)
        plt.ylabel('Flux')
        plt.xlabel('Time')
        plt.savefig(fpathout+'xyb_plot')


        send_mail('xyb_plot.png', fpathout, aor)

    return None


def backgr(a):
    
    backmask=np.zeros(shape=(32,32))
    backmask[10:20, 10:20]=1
    mean, median, std = sigma_clipped_stats(a, sigma=5.0, mask=backmask)

    return median

def centroid(a, cent):
    
    top=17
    bot=14
    a=ma.masked_invalid(a)
    #a=sigma_clip(a, sigma=7, iters=1)    
    a=a[bot:top+1, bot:top+1]
    a2=np.multiply(a,a)
    beta_np=np.sum(a)**2/np.sum(a2)
    if cent=='on':
        xc, yc = centroid_2dg(a)+bot
    else:
        xc, yc = centroid_com(a)+bot
    return (xc,yc)    

# def get_aor(plnm, plnm2, t1, basepath):
#     #a=find_all_dir('/Users/Brian/Desktop/Tucker_Group/Spitzer/'+plnm2)
#     if t1 != 'true':
#         #a=os.listdir('/Users/Brian/Desktop/Tucker_Group/Spitzer/'+plnm)
#         basepath='/Users/Brian/Desktop/Tucker_Group/Spitzer/'+plnm+'/'
#         a=glob.glob(basepath+ 'r*')
        

      
#     else: 
#         #basepath='/Users/Brian/Desktop/Tucker_Group/t_1/'
#         a=glob.glob(basepath+ 'r*')
#         nn=[ name for name in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, name)) ]
#         del nn[0]    

        
#     print(a)
#     sys.exit()
#     return a
def pmap_corr(position,  channel):


    from scipy.interpolate import griddata

    fpath='/Users/Brian/Documents/Python_Scripts/Spitzer/pmap_fits/'
    if channel == '1':
        hdulist=fits.open(fpath+'xgrid_ch1_500x500_0043_120828.fits')
        xgrid=hdulist[0].data
        hdulist.close()
        hdulist=fits.open(fpath+'ygrid_ch1_500x500_0043_120828.fits')
        ygrid=hdulist[0].data
        hdulist.close()
        hdulist=fits.open(fpath+'pmap_ch1_500x500_0043_120828.fits')
        pmap=hdulist[0].data
        hdulist.close()
    
    if channel=='2':
        hdulist=fits.open(fpath+'xgrid_ch2_0p1s_x4_500x500_0043_120124.fits')
        xgrid=hdulist[0].data
        hdulist.close()
        hdulist=fits.open(fpath+'ygrid_ch2_0p1s_x4_500x500_0043_120124.fits')
        ygrid=hdulist[0].data
        hdulist.close()
        hdulist=fits.open(fpath+'pmap_ch2_0p1s_x4_500x500_0043_120124.fits')
        pmap=hdulist[0].data
        hdulist.close()


    x=np.ravel(xgrid)
    y=np.ravel(ygrid)
    values=np.ravel(pmap)
    points=np.transpose(np.array([x,y]))
    if position.shape[0] != 2: position=np.transpose(position)
    corr=np.zeros(position.shape[1])
  
    corr=griddata(points, values, np.transpose(position), method='linear')
    
    return corr

def aor_from_list(planet, ch, basepath):
    filename=basepath+'aor_list.csv'
    #filename='/home/bkilpatr/Spitzer_Routines/aor_list.csv'
    aor_arr=pd.read_csv(filename)
    chlist=aor_arr.CH
    aorlist=aor_arr.AOR
    namelist=aor_arr.Target
    ch_aors=np.where((namelist==planet) & (chlist==ch))
    aor_selected=np.array(aorlist[np.squeeze(ch_aors)])


    return aor_selected

def send_mail(filename, filepath, aor):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders



    fromx = 'brian.m.kilpatrick@gmail.com'
    to  = 'brian_kilpatrick@brown.edu'
    msg = MIMEMultipart()
    #msg = MIMEText('Python test')
    msg['Subject'] = 'Extraction Complete: '+aor
    msg['From'] = fromx
    msg['To'] = to


    body='The extraction for AOR '+aor+' is now complete.  Plots are attached.\n'

    msg.attach(MIMEText(body, 'plain'))
    #filename="t1_sys.png"
    #filepath="/home/bkilpatr/mapping_files/"
    attachment= open(filepath+filename, 'rb')
    p = MIMEBase('application', 'octet-stream')
     
    # To change the payload into encoded form
    p.set_payload((attachment).read())
     
    # encode into base64
    encoders.encode_base64(p)
      
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
     
    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.ehlo()
    server.login('brian.m.kilpatrick@gmail.com', 'leXi518ann')
    server.sendmail(fromx, to, msg.as_string())
    server.quit()

    return None









#main(sys.argv[1], int(sys.argv[2]), [sys.argv[3]])
