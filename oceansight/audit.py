"""Train-only unsupervised quality review; scores are review priorities, not labels."""
import json
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from .common import DATA, REPORTS, setup, save_json

def features(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[8,4],[0,180,0,256]).flatten()
    hist/=max(hist.sum(),1)
    quality=[gray.mean()/255, gray.std()/255, np.log1p(cv2.Laplacian(gray,cv2.CV_64F).var()),
             np.mean(gray<20),np.mean(gray>235),hsv[:,:,1].mean()/255]
    return np.r_[hist,quality]

def audit():
    setup()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='train']
    images=[cv2.imread(str(DATA/'images'/'train'/r['file_name'])) for r in records]
    X=np.asarray([features(im) for im in images])
    scaled=StandardScaler().fit_transform(X)
    coords=PCA(n_components=2,random_state=42).fit_transform(scaled)
    clusters=KMeans(n_clusters=5,random_state=42,n_init=10).fit_predict(scaled)
    scores=-IsolationForest(n_estimators=200,contamination=.08,random_state=42).fit(scaled).score_samples(scaled)
    rows=[{'file_name':r['file_name'],'video':r['group'],'cluster':int(clusters[i]),'outlier_score':float(scores[i]),
           'pca_x':float(coords[i,0]),'pca_y':float(coords[i,1]),'brightness':float(X[i,-6]),'contrast':float(X[i,-5]),'log_sharpness':float(X[i,-4])} for i,r in enumerate(records)]
    df=pd.DataFrame(rows).sort_values('outlier_score',ascending=False)
    df.to_csv(REPORTS/'quality.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(coords[:,0],coords[:,1],c=clusters,cmap='tab10',s=20,alpha=.7)
    ax.set(title='Training images: color/texture/quality feature clusters',xlabel='PCA 1',ylabel='PCA 2')
    fig.tight_layout();fig.savefig(REPORTS/'quality_clusters.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(12,8))
    for ax,i in zip(axes.flat,np.argsort(scores)[-12:][::-1]):
        ax.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB));ax.set_title(f"{records[i]['file_name']}\nscore {scores[i]:.3f}",fontsize=7);ax.axis('off')
    fig.tight_layout();fig.savefig(REPORTS/'outlier_contact_sheet.jpg',dpi=130);plt.close(fig)
    save_json(REPORTS/'audit.json',{'images':len(records),'fit_split':'train only','features':'32-bin HSV histogram + brightness, contrast, log Laplacian sharpness, dark/bright fractions, saturation',
              'pipeline':'StandardScaler -> KMeans(5) and IsolationForest; PCA(2) visualization only',
              'interpretation':'Unsupervised appearance clusters and review scores, not semantic classes or verified label errors. No images removed.',
              'top_review_candidates':df.head(12).to_dict(orient='records')})

if __name__=='__main__': audit()
