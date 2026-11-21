"""Render a simple animated methodology diagram and a separately trained baseline rollout."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import io
from functools import lru_cache
import matplotlib
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont, ImageOps
from matplotlib.font_manager import findfont

CHAPTERS = [
    (0, 20, 'Simulator-trained policy rollout in simulation', 'Separate simulator-trained baseline 1199; this footage is not an online world-model training result.'),
    (20, 29, '1. Collect transitions and store them in a buffer', 'Policy actions produce simulator transitions: state, action, reward, next state, done.'),
    (29, 40, '2. Update the world model', 'Sample sequences, warm up the GRU, predict future states, and minimize prediction loss.'),
    (40, 49, '3. Train the policy in the simulator first', 'Before the switch: simulator rollouts provide the rewards and transitions for PPO.'),
    (49, 59, '3. Switch policy training to the world model', 'After warm-start: imagined transitions and custom rewards drive PPO updates.'),
    (59, 67, '4. Repeat world-model and policy optimization', 'Collect fresh transitions, update the world model, then update the policy again.'),
]
W,H=1280,720
INK='#263238'; GRAY='#65717a'; BLUE='#dae8fc'; GREEN='#d5e8d4'; YELLOW='#fff2cc'; PURPLE='#e1d5e7'; ACCENT='#2879b9'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout',type=Path,required=True)
    parser.add_argument('--checkpoint-label',default='Simulator baseline / checkpoint 1199')
    parser.add_argument('--output',type=Path,default=Path('docs/assets/videos/pipeline.mp4'))
    parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if args.output.exists() and not args.overwrite:raise FileExistsError(args.output)
    if args.fps<1:raise ValueError('fps must be positive')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    regular=findfont('DejaVu Sans');heavy=findfont('DejaVu Sans:weight=bold')
    fonts={n:ImageFont.truetype(regular,n) for n in [16,18,20,22,24,26,30,34]}
    bold={n:ImageFont.truetype(heavy,n) for n in [18,22,26,30,34]}
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    latex_labels = {
        'Policy': r'\mathrm{Policy}\;\pi_\theta',
        'Simulator': r'\mathrm{Simulator}',
        'Transition': r'\mathrm{Transition}',
        '(s, a, r, s′, done)': r'(s_t,a_t,r_t,s_{t+1},d_t)',
        'Replay buffer': r'\mathrm{Replay\ buffer}\;\mathcal{D}',
        'GRU world model': r'\mathrm{World\ model}\;f_\phi',
        'World model': r'\mathrm{World\ model}\;f_\phi',
        'Prediction loss': r'\mathrm{Loss}\;\mathcal{L}_{\mathrm{WM}}',
        'prediction vs. actual': r'\hat{s}_{t+1}\;\mathrm{vs.}\;s_{t+1}',
        'sample a sequence': r'(s_{t:t+M+N},a_{t:t+M+N})',
        'choose an action': r'a_t\sim\pi_\theta(\cdot\mid o_t)',
        'sample action': r'a_t\sim\pi_\theta(\cdot\mid o_t)',
        'next observation': r'o_{t+1}',
        'apply action': r'a_t\longrightarrow s_{t+1}',
        'store sequences': r'\mathcal{D}\leftarrow\mathcal{D}\cup\{\tau\}',
        'predict next state': r'\hat{s}_{t+1}=f_\phi(s_t,a_t,h_t)',
        'backpropagation + Adam': r'\nabla_\phi\mathcal{L}_{\mathrm{WM}}\;\longrightarrow\;\mathrm{Adam}',
        'GRU': r'\mathrm{GRU}',
        'feed prediction back': r'\hat{s}_{t+1}\;\longrightarrow\;\mathrm{next\ input}',
        'Observed history (M = 32)': r'\mathrm{Observed\ history}\;(M=32)',
        'Autoregressive forecast (N = 8)': r'\mathrm{Autoregressive\ forecast}\;(N=8)',
        'PPO optimizer': r'\mathrm{PPO}\;\mathcal{L}_{\pi}+\mathcal{L}_{V}',
        'update actor + critic': r'\theta,\psi\leftarrow\mathrm{PPO}(\theta,\psi)',
        'updated policy parameters': r'\mathrm{Updated\ policy}\;\pi_{\theta_{k+1}}',
        'rollouts + rewards': r'\tau,\;r_t,\;\hat{A}_t',
        'Update WM': r'\mathrm{Update}\;f_\phi',
        'Update policy': r'\mathrm{Update}\;\pi_\theta',
        'Observation': r'\mathrm{Observation}\;o_t',
        'Trained actor': r'\mathrm{Actor}\;\pi_\theta',
        'robot state + command': r'o_t=(s_t,c_t)',
        'choose joint targets': r'a_t=\pi_\theta(o_t)',
    }
    @lru_cache(maxsize=256)
    def formula(value, size, color):
        expression = latex_labels.get(value)
        if expression is None:
            expression = value if value.startswith('t = ') else r'\mathrm{' + value.replace(' ', r'\ ') + '}'
        buffer = io.BytesIO()
        with matplotlib.rc_context({'savefig.transparent': True}):
            math_to_image('$'+expression+'$', buffer, prop=FontProperties(size=size*72/120),
                          dpi=360, format='png', color=color)
        buffer.seek(0)
        rendered = Image.open(buffer).convert('RGBA')
        return rendered.resize((max(1,round(rendered.width/3)), max(1,round(rendered.height/3))), Image.Resampling.LANCZOS)
    reader=imageio.get_reader(args.rollout);source_fps=reader.get_meta_data()['fps'];source_count=reader.count_frames()
    writer=imageio.get_writer(args.output,fps=args.fps,codec='libx264',quality=8,macro_block_size=16)
    stills={4:'overview',10:'deployment',27:'collection',36:'world-model',45:'warm-start',55:'ppo',64:'loop'}
    try:
        for frame_id in range(67*args.fps):
            t=frame_id/args.fps
            chapter_index=next(i for i,(a,b,_,_) in enumerate(CHAPTERS) if a<=t<b)
            start,end,title,caption=CHAPTERS[chapter_index];local=t-start
            index=(chapter_index-1)%len(CHAPTERS)
            im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
            def text(x,y,value,size=22,color=INK,strong=False):
                if 140 <= y < 640:
                    label=formula(value,size,color)
                    if label.width > W-x-35:
                        label=ImageOps.contain(label,(W-x-35,label.height))
                    im.paste(label,(x,y+3),label)
                else:
                    d.text((x,y),value,font=(bold[size] if strong else fonts[size]),fill=color)
            def center(x,y,w,value,size=22,strong=False,color=INK):
                label=formula(value,size,color)
                if label.width > w-24 or label.height > 26:
                    label=ImageOps.contain(label,(w-24,26))
                im.paste(label,(x+(w-label.width)//2,y),label)
            def box(x,y,w,label,sub='',fill=BLUE,active=False):
                d.rounded_rectangle((x,y,x+w,y+86),radius=7,fill=fill,outline=ACCENT if active else '#555555',width=4 if active else 2)
                center(x,y+18,w,label,23,True)
                if sub:center(x,y+52,w,sub,19,color=GRAY)
            def line(points,phase=0,enabled=True):
                color=INK if enabled else '#d7dadd';d.line(points,fill=color,width=2)
                a,b=points[-2:];theta=math.atan2(b[1]-a[1],b[0]-a[0]);r=12
                d.polygon([b,(b[0]-r*math.cos(theta-.45),b[1]-r*math.sin(theta-.45)),(b[0]-r*math.cos(theta+.45),b[1]-r*math.sin(theta+.45))],fill=color)
                if not enabled:return
                lengths=[math.dist(a,b) for a,b in zip(points,points[1:])];distance=((local*.36+phase)%1)*sum(lengths)
                for (a,b),length in zip(zip(points,points[1:]),lengths):
                    if distance<=length:
                        f=distance/length;x=a[0]+f*(b[0]-a[0]);y=a[1]+f*(b[1]-a[1]);d.ellipse((x-6,y-6,x+6,y+6),fill=ACCENT);break
                    distance-=length
            text(45,26,'Online world-model learning / intended methodology',18,GRAY)
            text(45,66,title,30,strong=True)
            d.line((45,121,1235,121),fill='#dfe3e7',width=1)
            if index==0:
                active=int(local/1.3)%4
                box(65,255,220,'Policy','choose an action',GREEN,active==0)
                box(410,255,230,'Simulator','apply action',BLUE,active==1)
                box(770,255,240,'Transition','(s, a, r, s′, done)',YELLOW,active==2)
                line([(285,298),(408,298)]);line([(640,298),(768,298)],.2)
                box(770,455,240,'Replay buffer','store sequences',PURPLE,active==3)
                line([(890,343),(890,453)],.4)
                line([(525,255),(525,185),(175,185),(175,253)],.6)
                text(259,151,'next observation',18,GRAY)
                count=1+int(local*2)%12
                for i in range(count):d.rectangle((1050,535-i*19,1150,548-i*19),fill=PURPLE,outline='#9673a6')
                line([(1010,498),(1048,498)],.8)
                text(65,565,'Collect real simulator experience throughout the online loop.',22)
            elif index==1:
                box(55,183,240,'Replay buffer','sample a sequence',PURPLE,int(local)%4==0)
                box(470,183,250,'GRU world model','predict next state',BLUE,int(local)%4==1)
                box(935,183,280,'Prediction loss','prediction vs. actual',YELLOW,int(local)%4==2)
                line([(295,226),(468,226)]);line([(720,226),(933,226)],.3)
                line([(1075,271),(1075,335),(595,335),(595,271)],.6)
                text(700,300,'backpropagation + Adam',20,GRAY)
                for i in range(5):
                    x=130+i*220;active=int(local*1.4)%5==i
                    box(x,427,125,'GRU',f't = {i}',PURPLE,active)
                    if i<4:line([(x+125,470),(x+218,470)],i*.13)
                text(105,547,'Observed history (M = 32)',20,GRAY)
                text(750,547,'Autoregressive forecast (N = 8)',20,GRAY)
                line([(1035,427),(1035,390),(1105,390),(1105,427)],.5)
                text(896,360,'feed prediction back',18,GRAY)
            elif index in [2,3]:
                using_wm=index==3
                box(65,282,245,'Policy','sample action',GREEN,int(local)%4==0)
                box(490,185,275,'Simulator','before warm-start ends',BLUE,not using_wm)
                box(490,385,275,'World model','after warm-start',PURPLE,using_wm)
                line([(310,325),(390,325),(390,228),(488,228)],enabled=not using_wm)
                line([(310,325),(390,325),(390,428),(488,428)],enabled=using_wm)
                box(950,282,270,'PPO optimizer','update actor + critic',YELLOW,int(local)%4==3)
                line([(765,228),(850,228),(850,325),(948,325)],.35,not using_wm)
                line([(765,428),(850,428),(850,325),(948,325)],.35,using_wm)
                text(940,240,'rollouts + rewards',18,GRAY)
                line([(310,325),(390,325)],.1)
                line([(850,325),(948,325)],.5)
                line([(1085,370),(1085,557),(187,557),(187,370)],.6)
                text(465,575,'updated policy parameters',20,GRAY)
                if using_wm:
                    text(60,151,'Warm-start complete: switch the source of policy-training transitions.',22)
                else:text(60,151,'Initial phase: learn from simulator transitions while fitting the world model.',22)
            elif index==4:
                labels=[('Collect transitions','refresh replay',BLUE),('Update WM','fit sampled sequences',PURPLE),('Update policy','PPO in imagination',GREEN)]
                active=int(local*1.0)%3
                for i,(name,sub,color) in enumerate(labels):box(90+i*410,270,280,name,sub,color,active==i)
                line([(370,313),(498,313)]);line([(780,313),(908,313)],.3)
                line([(1050,358),(1050,490),(230,490),(230,358)],.6)
                text(420,525,'Repeat with the improved policy',26)
                text(90,175,'After the warm-start switch',22,GRAY)
            else:
                source_index=min(int(local*source_fps),source_count-1)
                video=Image.fromarray(reader.get_data(source_index));video=ImageOps.contain(video,(850,470))
                im.paste(video,(35+(850-video.width)//2,145+(470-video.height)//2))
                box(935,188,280,'Observation','robot state + command',BLUE)
                box(935,338,280,'Trained actor','choose joint targets',GREEN)
                line([(1075,276),(1075,336)])
                line([(1075,426),(1075,500),(887,500)],.4)
                text(942,530,'MuJoCo simulation',20,GRAY)
                text(45,609,args.checkpoint_label+' | command 0.5 m/s',18)
            d.line((45,647,1235,647),fill='#dfe3e7',width=1)
            words=caption.split();row='';y=660
            for word in words:
                trial=(row+' '+word).strip()
                if d.textlength(trial,font=fonts[20])>1180:text(45,y,row,20);row=word;y+=25
                else:row=trial
            text(45,y,row,20)
            d.rectangle((0,716,int(W*t/67),719),fill=ACCENT)
            writer.append_data(np.asarray(im))
            for at,name in stills.items():
                if frame_id==at*args.fps:im.save(args.output.with_name('pipeline-'+name+'.png'))
            if frame_id==start*args.fps:print(title,flush=True)
    finally:reader.close();writer.close()
    def ts(s):return f'00:{int(s)//60:02d}:{int(s)%60:02d},000'
    args.output.with_suffix('.srt').write_text('\n\n'.join(f'{i+1}\n{ts(a)} --> {ts(b)}\n{title}\n{caption}' for i,(a,b,title,caption) in enumerate(CHAPTERS))+'\n')
    with args.rollout.open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps({'duration_seconds':67,'fps':args.fps,'resolution':[W,H],'audio':'none; burned-in captions','rollout':os.path.relpath(args.rollout),'rollout_sha256':checksum,'checkpoint_label':args.checkpoint_label,'chapters':CHAPTERS,'methodology_note':'Requested intended sequence includes simulator PPO before switching to WM PPO. Current online implementation collects data and fits WM during warm-start but does not yet perform those initial simulator PPO updates.','footage_note':'Separately trained simulator baseline, not a policy produced by the illustrated WM training run.'},indent=2)+'\n')

if __name__=='__main__':main()
