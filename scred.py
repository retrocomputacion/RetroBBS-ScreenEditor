import dearpygui.dearpygui as dpg
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageOps
from os.path import exists as file_exists

chars_msx = None    # Charset texture array
chars_cbmu = None
chars_cbml = None
raw_tex = None  # About splash texture
dummy_tex = None
pimg = None     # About splash image
iimg = None     # Icons image
Quit = False
screen = Image.new(mode='RGB',size=(640,400),color=(0,0,0)) # Main screen image
raw_screen = np.asarray(screen, dtype=np.float32)/255       # Main screen texture
raw_over = np.full((400,640,4),[0,0,0,0], dtype=np.float32)   # Overlay texture
sdraw = ImageDraw.Draw(screen)
prev_tex = np.zeros((240,200,3), dtype=np.float32)  # Open file dialog preview texture
raw_tile = None # Current tile texture
raw_prev = None # Current tile preview texture
raw_icons = []  # Icon textures
raw_icons16 = [] # Small icon textures
curchar = 32
colors = [0,14]     # Bg/Fg (-1)
background = 0      # Initial Background color 
border = 0          # Initial Border color
matrix = np.full((32,24,3),[32,0,14])    #[[0,[0,14]]]*768   # Tile/Bg/Fg (-1)
tmpmat = None       # Temporal Undo matrix
change = False
grid_en = True
over_en = False
old_c = [0,0]
old_m = np.array([0,-1,-1])
mode = 0            # Mode: 0 = Draw 1 = Flood fill 2 = Text 3 = Recolor 4 = Select 5 = Brush 6 = Color pick
p_mode = 0          # Previous mode
text_c = [0,0]

charset = 'Upper/GFX'   #CBM charset

modes = {'MSX Screen 2 - RetroTerm':{'geo':(32,24),'palette':[[0x00,0x00,0x00],[0x0a,0xad,0x1e],[0x34,0xc8,0x4c],[0x2b,0x2d,0xe3],[0x51,0x4b,0xfb],[0xbd,0x29,0x25],[0x1e,0xe2,0xef],[0xfb,0x2c,0x2b],
         [0xff,0x5f,0x4c],[0xbd,0xa2,0x2b],[0xd7,0xb4,0x54],[0x0a,0x8c,0x18],[0xaf,0x32,0x9a],[0xb2,0xb2,0xb2],[0xff,0xff,0xff]],'default':[32,0,14],
         'filetypes':['MSX sequential','.mseq']},
         'Commodore 64':{'geo':(40,25),'palette':[[0x00,0x00,0x00],[0xff,0xff,0xff],[0x7C,0x35,0x2B],[0x5A,0xA6,0xB1],[0x69,0x41,0x85],[0x5D,0x86,0x43],[0x21,0x2E,0x78],[0xCF,0xBE,0x6F],
         [0x89,0x4A,0x26],[0x5B,0x33,0x00],[0xAF,0x64,0x59],[0x43,0x43,0x43],[0x6b,0x6b,0x6b],[0xA0,0xCB,0x84],[0x56,0x65,0xB3],[0x95,0x95,0x95]],'default':[32,0,1],
         'filetypes':['Sequential','.seq']}
         }

petcolors = [b'\x90',b'\x05',b'\x1c',b'\x9f',b'\x9c',b'\x1e',b'\x1f',b'\x9e',b'\x81',b'\x95',b'\x96',b'\x97',b'\x98',b'\x99',b'\x9a',b'\x9b']


# filetype = ['MSX sequential','.mseq']

screen_mode = 'MSX Screen 2 - RetroTerm'

undobuffer = deque(maxlen=10)
redobuffer = deque(maxlen=10)

clipboard = [None,None,[-1,-1]]

##### MSX Palette #####
palette =     [[0x00,0x00,0x00],[0x0a,0xad,0x1e],[0x34,0xc8,0x4c],[0x2b,0x2d,0xe3],[0x51,0x4b,0xfb],[0xbd,0x29,0x25],[0x1e,0xe2,0xef],[0xfb,0x2c,0x2b],
               [0xff,0x5f,0x4c],[0xbd,0xa2,0x2b],[0xd7,0xb4,0x54],[0x0a,0x8c,0x18],[0xaf,0x32,0x9a],[0xb2,0xb2,0xb2],[0xff,0xff,0xff]]



####### CALLBACKS #######
def quit_callback():
    global Quit
    
    dpg.configure_item("quit_id", show=False)
    Quit = True

# Show file->open dialog
def show_dialog():
    dpg.show_item("click_handler")
    dpg.show_item("open_dialog")

def print_me(sender, app_data):
    print(f"Menu Item: {sender} - {app_data}")

def draw_tile():
    global raw_screen
    
    x = int((dpg.get_mouse_pos()[0]-30)//16)*16
    y = int((dpg.get_mouse_pos()[1]-30)//16)*16
    # sdraw.rectangle([(x,y),(x+15,y+15)],fill=(255,0,0))
    raw_screen[y:y+16, x:x+16] = raw_tile

def select_char(sender):
    global curchar, raw_tile, raw_prev
    chars = chars_msx if screen_mode == 'MSX Screen 2 - RetroTerm' else chars_cbmu if charset == 'Upper/GFX' else chars_cbml
    pre = sender[0] #'b' if screen_mode == 'MSX Screen 2 - RetroTerm' else 'd'
    dpg.configure_item(pre+str(curchar),tint_color=(128,128,128))
    curchar = int(sender[1:])
    dpg.configure_item('prevcode',default_value=f'0x{curchar:02x}')
    raw_tile[np.where((chars[curchar]==[0,0,0]).all(axis=2))] = np.array(palette[colors[0]],dtype=np.float32)/255
    raw_tile[np.where((chars[curchar]==[1,1,1]).all(axis=2))] = np.array(palette[colors[1]],dtype=np.float32)/255
    tmp_tile = np.asarray(Image.fromarray((raw_tile*255).astype('uint8')).resize([32,32], resample=Image.NEAREST), dtype=np.float32)/255
    raw_prev[:] = tmp_tile[:]
    dpg.configure_item(sender,tint_color=(255,255,128))
    if mode > 1:
        set_mode('draw_b')

def clear_screen():
    global matrix, raw_screen, undobuffer, redobuffer, background

    undobuffer.append(matrix)
    redobuffer.clear()
    matrix = np.full((modes[screen_mode]['geo'][0],modes[screen_mode]['geo'][1],3),[modes[screen_mode]['default'][0],background,colors[1]]) #[[0,colors.copy()]]*768
    # background = colors[0]
    raw_screen[:,:] = np.array(palette[background],dtype=np.float32)/255
    # dpg.set_value('color2', palette[colors[0]])

def flood_fill():
    global matrix, undobuffer, redobuffer

    if mode == 1:
        col = int((dpg.get_mouse_pos()[0]-32)//16)
        row = int((dpg.get_mouse_pos()[1]-32)//16)
        test = matrix[col,row].copy()
        stack = [(col,row)]
        if np.array_equal(matrix[col,row],[curchar,colors[0],colors[1]]):
            return

        mc = modes[screen_mode]['geo'][0]-1
        mr = modes[screen_mode]['geo'][1]-1
        undobuffer.append(matrix.copy())
        redobuffer.clear()
        while stack:
            col,row = stack.pop(0)
            if np.array_equal(matrix[col,row],test):
                matrix[col,row,:] = [curchar,colors[0],colors[1]]
                if col > 0:
                    stack.append((col-1,row))
                if col < mc:
                    stack.append((col+1,row))
                if row > 0:
                    stack.append((col,row-1))
                if row < mr:
                    stack.append((col,row+1))
        sync_matrix()

def grid():
    global grid_en
    if grid_en:
        dpg.configure_item('grid', show=False)
        dpg.configure_item('grid_b', tint_color=(128,128,128))
        grid_en = False
    else:
        dpg.configure_item('grid', show=True)
        dpg.configure_item('grid_b', tint_color=(255,255,255))
        grid_en = True

def overlay():
    global over_en
    if over_en:
        dpg.configure_item('overlay', show=False)
        dpg.configure_item('over_b', tint_color=(128,128,128))
        over_en = False
    else:
        dpg.configure_item('overlay', show=True)
        dpg.configure_item('over_b', tint_color=(255,255,255))
        over_en = True

def show_about():
    cfg = dpg.get_item_configuration('MainW')
    dpg.configure_item("aboutw_id", pos=((cfg['width']/2)-180, (cfg['height']/2)-120))
    dpg.configure_item("aboutw_id", show = True)

def about_callback():
    dpg.configure_item("aboutw_id", show = False)

def set_color(sender,appdata):
    global colors, raw_tile, raw_prev, background, matrix, border
    if type(sender) == str and sender[0] in ['2','3']:
        if sender[0] == '2':
            background = int(sender[7:])
        else:
            border = int(sender[7:])
        if screen_mode == 'Commodore 64' and sender[0] == '2':
            colors[0] = background
            pre = 'd' if charset == 'Upper/GFX' else 'e'
            select_char(pre+str(curchar))
            matrix[...,1] = background
            sync_matrix()
        dpg.set_value('color'+sender[0], dpg.get_value(sender))
        dpg.configure_item('cselect_id', show=False)
        dpg.delete_item('cselect_id')
    else:
        chars = chars_msx if screen_mode == 'MSX Screen 2 - RetroTerm' else chars_cbmu if charset == 'Upper/GFX' else chars_cbml
        sender = dpg.get_item_alias(appdata[1])
        c = int(not appdata[0])
        if c == 0 and screen_mode == 'Commodore 64':
            return
        colors[c] = int(sender[6:])
        raw_tile[np.where((chars[curchar]==[0,0,0]).all(axis=2))] = np.array(palette[colors[0]],dtype=np.float32)/255
        raw_tile[np.where((chars[curchar]==[1,1,1]).all(axis=2))] = np.array(palette[colors[1]],dtype=np.float32)/255
        tmp_tile = np.asarray(Image.fromarray((raw_tile*255).astype('uint8')).resize([32,32], resample=Image.NEAREST), dtype=np.float32)/255
        raw_prev[:] = tmp_tile[:]
        dpg.set_value('color'+str(c), dpg.get_value(sender))

def swap_colors():
    global colors, raw_tile, raw_prev

    chars = chars_msx if screen_mode == 'MSX Screen 2 - RetroTerm' else chars_cbmu if charset == 'Upper/GFX' else chars_cbml
    colors = colors[::-1]   #[colors[1],colors[0]]
    dpg.set_value('color0', palette[colors[0]])
    dpg.set_value('color1', palette[colors[1]])
    raw_tile[np.where((chars[curchar]==[0,0,0]).all(axis=2))] = np.array(palette[colors[0]],dtype=np.float32)/255
    raw_tile[np.where((chars[curchar]==[1,1,1]).all(axis=2))] = np.array(palette[colors[1]],dtype=np.float32)/255
    tmp_tile = np.asarray(Image.fromarray((raw_tile*255).astype('uint8')).resize([32,32], resample=Image.NEAREST), dtype=np.float32)/255
    raw_prev[:] = tmp_tile[:]

def color_selector(sender):
    color_count = len(palette)
    ix = int(sender[5:])
    wpos = (200,300) if ix in [2,3] else (700,300)
    with dpg.window(tag="cselect_id", show=True, no_close=True, modal=True,pos=wpos):
        with dpg.group(tag='c_select'):
                for j in range(0,color_count,8):
                    with dpg.group(horizontal=True):
                        for k in range(j,j+8 if j+8<color_count else color_count):
                            dpg.add_color_button(palette[k],tag=str(ix)+'bcolor'+str(k), callback=set_color, no_alpha=True)

# Undo/Redo handler
def undoredo(sender):
    global change, redobuffer, undobuffer, matrix

    if (sender == 'undo') and (len(undobuffer) > 0):
        redobuffer.append(matrix)
        matrix = undobuffer.pop()
        change = False
        sync_matrix()
    elif len(redobuffer) > 0:
        undobuffer.append(matrix)
        matrix = redobuffer.pop()
        change = False
        sync_matrix()

def flip(sender):
    global clipboard
    tmp = np.zeros(clipboard[1].shape, dtype=np.float32)
    if sender == 'flip_y':
        clipboard[0] = np.fliplr(clipboard[0])
        for i in range(0,tmp.shape[0],16):
            tmp[tmp.shape[0]-i-16:tmp.shape[0]-i] = clipboard[1][i:i+16]
        clipboard[1][:] = tmp[:]    #np.flipud(clipboard[1])[:]
    else:
        # clipboard[1][:] = np.fliplr(clipboard[1])[:]
        for i in range(0,tmp.shape[1],16):
            tmp[:,tmp.shape[1]-i-16:tmp.shape[1]-i] = clipboard[1][:,i:i+16]
        clipboard[1][:] = tmp[:]
        clipboard[0] = np.flipud(clipboard[0])

def set_mode(sender):
    global mode
    dmodes = ['draw_b','fill_b','text_b','paint_b','select_b']
    if sender in dmodes:
        mode = dmodes.index(sender)
        for b in dmodes:
            tint = (255,255,255) if b == sender else (128,128,128)
            dpg.configure_item(b,tint_color=tint)
        dpg.configure_item('brush_b', tint_color=(128,128,128))
        dpg.configure_item('flip', show=False)
        dpg.configure_item('text_handler', show=mode==2)
        dpg.configure_item('cursor', show=mode==2)
        dpg.configure_item('selection', show=mode==4, pmin=(0,0), pmax=(0,0))
        if clipboard[0] is not None:
            dpg.configure_item('clip', show=False)
    elif sender == 'brush_b' and clipboard[0] is not None:
        mode = 5
        for b in dmodes:
            dpg.configure_item(b,tint_color=(128,128,128))
        dpg.configure_item('brush_b', tint_color=(255,255,255))
        dpg.configure_item('flip', show=True)
        dpg.configure_item('text_handler', show=False)
        dpg.configure_item('cursor', show=False)
        dpg.configure_item('selection', show=False, pmin=(0,0), pmax=(0,0))
        if clipboard[0] is not None:
            dpg.configure_item('clip', show=True)
        clipboard[2] = [-1,-1]

def set_screenmode(sender):
    global screen_mode,colors,background,over_en,raw_over,palette, clipboard
    smode = dpg.get_value('s_mode')
    if smode != screen_mode:
        pre = 'b' if screen_mode == 'MSX Screen 2 - RetroTerm' else 'd' if charset == 'Upper/GFX' else 'e'
        dpg.configure_item(pre+str(curchar),tint_color=(128,128,128))   # Reset char selection for previous mode
        screen_mode = smode
        palette = modes[screen_mode]['palette']
        dpg.set_item_width('mainimg',modes[screen_mode]['geo'][0]*16)
        dpg.set_item_height('mainimg',modes[screen_mode]['geo'][1]*16)
        if screen_mode == "Commodore 64":
            dpg.configure_item('cbmu_chars', show=True)
            dpg.configure_item('msx_chars', show=False)
            dpg.configure_item('swap', show=False)
            dpg.configure_item('color0', show=False)
            dpg.configure_item('charset', show=True)
            dpg.configure_item('c64colors', show=True)
            dpg.configure_item('msxcolors', show=False)
        else:
            dpg.configure_item('cbmu_chars', show=False)
            dpg.configure_item('msx_chars', show=True)
            dpg.configure_item('swap', show=True)
            dpg.configure_item('color0', show=True)
            dpg.configure_item('charset', show=False)
            dpg.configure_item('c64colors', show=False)
            dpg.configure_item('msxcolors', show=True)
        dpg.configure_item('overlay', show=False)
        dpg.configure_item('over_b', tint_color=(128,128,128))
        # dpg.set_value('color2', palette[colors[0]])
        over_en = False
        raw_over.fill(0)
        background = modes[screen_mode]['default'][1]
        colors = modes[screen_mode]['default'][1:]
        dpg.set_value('color0', palette[colors[0]])
        dpg.set_value('color1', palette[colors[1]])
        dpg.set_value('color2', palette[colors[0]])
        dpg.set_value('color3', palette[colors[0]])
        pre = 'b' if screen_mode == 'MSX Screen 2 - RetroTerm' else 'd' if charset == 'Upper/GFX' else 'e'
        select_char(pre+str(modes[screen_mode]['default'][0]))
        if clipboard[0] is not None:
            dpg.delete_item('clip')     #Delete clip draw image
            dpg.delete_item('clip_t')   #Delete clip texture
        clipboard = [None,None,[-1,-1]]
        new_work()

def set_charset(sender, app_data):
    global charset, clipboard
    if charset != app_data:
        dpg.configure_item('cbmu_chars', show= not dpg.get_item_state('cbmu_chars')['visible'])
        dpg.configure_item('cbml_chars', show= not dpg.get_item_state('cbml_chars')['visible'])
        pre = 'd' if charset == 'Upper/GFX' else 'e'
        dpg.configure_item(pre+str(curchar),tint_color=(128,128,128))   # Reset char selection for previous mode
        charset = app_data
        pre = 'd' if charset == 'Upper/GFX' else 'e'
        select_char(pre+str(modes[screen_mode]['default'][0]))
        sync_matrix()
        if clipboard[0] is not None:
            dpg.delete_item('clip')     #Delete clip draw image
            dpg.delete_item('clip_t')   #Delete clip texture
        clipboard = [None,None,[-1,-1]]

# Process mouse clicks for file dialog image preview
def fileclick(sender, app_data):
    global prev_tex
    data = dpg.get_file_dialog_info("open_dialog")
    u_data = dpg.get_item_user_data("open_dialog")
    if data['file_path_name']!=u_data:
        dpg.set_item_user_data("open_dialog", data['file_path_name'])
        try:    #Try loading the image preview
            tmp = Image.open(data['file_path_name']).convert('RGB')
            pv_img= ImageOps.pad(tmp,(200,240))
            tmp = np.asarray(pv_img, dtype=np.float32)/255
            prev_tex[:] = tmp[:]    #copy input image to Dear PyQUI buffer
        except:
            prev_tex.fill(0)

def show_save(sender, app_data):
    # Save file dialog
    filetype = modes[screen_mode]['filetypes']
    with dpg.file_dialog(label="Save "+filetype[0]+" file", directory_selector=False, show=True, callback=check_file, min_size=(500,400), user_data=filetype):
        dpg.add_file_extension("", color=(150, 255, 150, 255))
        dpg.add_file_extension(extension=filetype[1], color=(255, 255, 64, 255))

def show_open(sender, app_data):
    # Open file dialog
    filetype = modes[screen_mode]['filetypes']
    with dpg.file_dialog(label="Open "+filetype[0]+" file", directory_selector=False, show=True, callback=open_seq, min_size=(500,400), user_data=filetype):
        dpg.add_file_extension("", color=(150, 255, 150, 255))
        dpg.add_file_extension(extension=filetype[1], color=(255, 255, 64, 255))


#########################

def drag_handler(sender, app_data):
    global raw_screen, matrix, tmpmat, change, old_c, old_m, text_c


    maxc = modes[screen_mode]['geo'][0]-1
    maxr = modes[screen_mode]['geo'][1]-1
    maxx = modes[screen_mode]['geo'][0]*16
    maxy = modes[screen_mode]['geo'][1]*16

    htype=dpg.get_item_info(sender)["type"]
    c = int((dpg.get_mouse_pos()[0]-32)//16)
    r = int((dpg.get_mouse_pos()[1]-32)//16)
    c = 0 if c < 0 else (c if c <= maxc else maxc)
    r = 0 if r < 0 else (r if r <= maxr else maxr)
    if dpg.is_item_active('mainimg') and mode == 0: #Draw
        if htype=="mvAppItemType::mvFocusHandler":
            x = c*16
            y = r*16
            cur = [curchar,colors[0],colors[1]]
            if not np.array_equal(matrix[c,r],cur):
                if not change:
                    tmpmat = matrix.copy()  # Make a copy of the matrix before it gets modified
                raw_screen[y:y+16, x:x+16] = raw_tile
                matrix[c,r] = cur
                change = True
                dpg.configure_item('char_hover',default_value=f'0x{matrix[c,r,0]:02x}')
                old_m = matrix[c,r]
                dpg.configure_item('bgcolor', default_value=palette[matrix[c,r,1]])
                dpg.configure_item('fgcolor', default_value=palette[matrix[c,r,2]])
    elif dpg.is_item_active('mainimg') and mode == 2:   #Text
        if htype=="mvAppItemType::mvFocusHandler":
            text_c = [c,r]
            dpg.configure_item('cursor',pmin=(text_c[0]*16,text_c[1]*16),pmax=(16+text_c[0]*16,16+text_c[1]*16))
    elif dpg.is_item_active('mainimg') and mode == 3:   #Recolor
        if htype=="mvAppItemType::mvFocusHandler":
            x = c*16
            y = r*16
            cur = [matrix[c,r,0],colors[0],colors[1]]
            if not np.array_equal(matrix[c,r],cur):
                if not change:
                    tmpmat = matrix.copy()  # Make a copy of the matrix before it gets modified
                chars = chars_msx if screen_mode == 'MSX Screen 2 - RetroTerm' else chars_cbmu if charset == 'Upper/GFX' else chars_cbml
                tchar = np.zeros((16,16,3),np.float32)
                tchar[np.where((chars[cur[0]]==[0,0,0]).all(axis=2))] = np.array(palette[colors[0]],dtype=np.float32)/255
                tchar[np.where((chars[cur[0]]==[1,1,1]).all(axis=2))] = np.array(palette[colors[1]],dtype=np.float32)/255

                raw_screen[y:y+16, x:x+16] = tchar
                matrix[c,r] = cur
                change = True
                dpg.configure_item('char_hover',default_value=f'0x{matrix[c,r,0]:02x}')
                old_m = matrix[c,r]
                dpg.configure_item('bgcolor', default_value=palette[matrix[c,r,1]])
                dpg.configure_item('fgcolor', default_value=palette[matrix[c,r,2]])
    elif dpg.is_item_active('mainimg') and mode == 4:   #Select
        if htype=="mvAppItemType::mvFocusHandler":
            x = (1+c)*16
            y = (1+r)*16
            min = dpg.get_item_configuration('selection')['pmin']
            if min[0] < x and min[1] < y:
                dpg.configure_item('selection', pmax=(x,y))
    elif dpg.is_item_active('mainimg') and mode == 5:   #Brush
        if htype=="mvAppItemType::mvFocusHandler" and (clipboard[2][0] != c or clipboard[2][1] != r):
            if not change:
                tmpmat = matrix.copy()  # Make a copy of the matrix before it gets modified
                change = True
            x = c*16
            y = r*16
            cw = clipboard[1].shape[1]
            ch = clipboard[1].shape[0]
            x1 = x-((cw//32)*16)
            y1 = y-((ch//32)*16)
            x2 = x1+cw
            y2 = y1+ch
            if x1 < 0:
                cx1 = -x1
                x1 = 0
            else:
                cx1 = 0
            if y1 < 0:
                cy1 = -y1
                y1 = 0
            else:
                cy1 = 0
            if x2 > maxx:
                cx2 = cw-(x2-maxx)
                x2 = maxx
            else:
                cx2 = cw
            if y2 > maxy:
                cy2 = ch-(y2-maxy)
                y2 = maxy
            else:
                cy2 = ch
            raw_screen[y1:y2,x1:x2] = clipboard[1][cy1:cy2,cx1:cx2]
            matrix[x1//16:x2//16,y1//16:y2//16] = clipboard[0][cx1//16:cx2//16,cy1//16:cy2//16]
            clipboard[2] = [c,r]

    if dpg.is_item_hovered('mainimg'):
        if old_c != [c,r] or old_m[0] != matrix[c,r,0]:
            dpg.configure_item('coords',default_value=f'x:{c} y:{r}')
            old_c = [c,r]
            dpg.configure_item('char_hover',default_value=f'0x{matrix[c,r,0]:02x}')
            dpg.configure_item('bgcolor', default_value=palette[matrix[c,r,1]])
            dpg.configure_item('fgcolor', default_value=palette[matrix[c,r,2]])
            old_m = matrix[c,r]
        if mode == 5:
            x = c*16
            y = r*16
            cw = clipboard[1].shape[1]
            ch = clipboard[1].shape[0]
            x1 = x-((cw//32)*16)
            y1 = y-((ch//32)*16)
            dpg.configure_item('clip',pmin=(x1,y1), pmax=(x1+cw,y1+ch))

def release_handler():
    global change, undobuffer, redobuffer, clipboard
    if change:
        undobuffer.append(tmpmat)
        redobuffer.clear()
        change = False
    if mode == 4:
        min = dpg.get_item_configuration('selection')['pmin']
        max = dpg.get_item_configuration('selection')['pmax']
        clipboard[1] = raw_screen[int(min[1]):int(max[1]),int(min[0]):int(max[0])].copy()
        min = [int(i//16) for i in min]
        max = [int(i//16) for i in max]
        clipboard[0] = matrix[min[0]:max[0],min[1]:max[1]].copy()
        with dpg.texture_registry():
            dpg.add_raw_texture(width=clipboard[1].shape[1], height=clipboard[1].shape[0], default_value=clipboard[1], tag='clip_t', format=dpg.mvFormat_Float_rgb)
        dpg.draw_image('clip_t',(0,0),(clipboard[1].shape[1],clipboard[1].shape[0]), uv_min=(0,0), uv_max=(1,1), tag='clip', parent='mainimg', before='overlay')
        set_mode('brush_b')

def click_handler(sender):
    if dpg.is_item_active('mainimg') and mode == 4: #Select
        if clipboard[0] is not None:
            dpg.delete_item('clip')     #Delete clip draw image
            dpg.delete_item('clip_t')   #Delete clip texture

        c = int((dpg.get_mouse_pos()[0]-32)//16)
        r = int((dpg.get_mouse_pos()[1]-32)//16)
        c = (0 if c < 0 else (c if c <= 31 else 31))*16
        r = (0 if r < 0 else (r if r <= 23 else 23))*16
        dpg.configure_item('selection', pmin=(c,r))

# Keyboard shortcuts handler
def short_handler(sender, data):
    type=dpg.get_item_info(sender)["type"]
    if type=="mvAppItemType::mvKeyPressHandler":
        if dpg.is_key_down(dpg.mvKey_Control):
            if data == ord('Z'):
                undoredo('undo')
            elif data == ord('Y'):
                undoredo('redo')
        elif dpg.is_key_down(dpg.mvKey_Alt):
            if data == ord('G'):    #Grid
                grid()
            elif data == ord('O'):  #Overlay
                overlay()
        elif dpg.is_key_down(dpg.mvKey_RShift) or dpg.is_key_down(dpg.mvKey_Shift):
            if data == 268:
                clear_screen()

# Text mode keyboard shortcuts handler:
def text_mode(sender, data):
    global text_c, raw_screen, undobuffer, redobuffer,mode

    chars = chars_msx if screen_mode == 'MSX Screen 2 - RetroTerm' else chars_cbmu if charset == 'Upper/GFX' else chars_cbml
    if not dpg.is_key_down(dpg.mvKey_Control) and not dpg.is_key_down(dpg.mvKey_Alt):
        mc = modes[screen_mode]['geo'][0]-1
        mr = modes[screen_mode]['geo'][1]-1
        def_char = modes[screen_mode]['default'][0]
        if data in range(0x20,0x7b):
            if not dpg.is_key_down(dpg.mvKey_RShift) and not dpg.is_key_down(dpg.mvKey_Shift):
                if data in range(0x41,0x5b):
                    data += 0x20
            elif data in range(0x31,0x3a):
                data -= 0x10
            if screen_mode == 'Commodore 64':   #Simple Ascii to Screencodes
                if data in range(0x41,0x60):
                    data -= 0x40
                elif data in range(0x61,0x7b):
                    data -= 0x20
            undobuffer.append(matrix.copy())
            redobuffer.clear()
            cur = [data,colors[0],colors[1]]
            matrix[text_c[0],text_c[1]] = cur
            tchar = np.zeros((16,16,3),np.float32)
            tchar[np.where((chars[data]==[0,0,0]).all(axis=2))] = np.array(palette[colors[0]],dtype=np.float32)/255
            tchar[np.where((chars[data]==[1,1,1]).all(axis=2))] = np.array(palette[colors[1]],dtype=np.float32)/255
            x = text_c[0]*16
            y = text_c[1]*16
            raw_screen[y:y+16, x:x+16] = tchar
            if text_c[0] == mc:
                if text_c[1] < mr:
                    text_c[0] = 0
                    text_c[1] += 1
            else:
                text_c[0] += 1
        elif data == 257:       # Enter
            dpg.configure_item('text_handler', show=False)
            mode = 0    # Return to drawing mode
            dpg.configure_item('draw_b', tint_color=(255,255,255))
            dpg.configure_item('text_b', tint_color=(128,128,128))
            dpg.configure_item('cursor', show=False)
        elif data == 259:       # Backspace
            undobuffer.append(matrix.copy())
            redobuffer.clear()
            if text_c[0] == 0:
                if text_c[1] > 0:
                    text_c[0] = mc
                    text_c[1] -= 1
            else:
                text_c[0] -= 1
            cur = [def_char,colors[0],colors[1]]
            matrix[text_c[0],text_c[1]] = cur
            tchar = np.zeros((16,16,3),np.float32)
            tchar[np.where((chars[def_char]==[0,0,0]).all(axis=2))] = np.array(palette[colors[0]],dtype=np.float32)/255
            tchar[np.where((chars[def_char]==[1,1,1]).all(axis=2))] = np.array(palette[colors[1]],dtype=np.float32)/255
            x = text_c[0]*16
            y = text_c[1]*16
            raw_screen[y:y+16, x:x+16] = tchar
        elif data == 262:       # Right crsr
            if text_c[0] == mc:
                if text_c[1] < mr:
                    text_c[0] = 0
                    text_c[1] += 1
            else:
                text_c[0] += 1
        elif data == 263:       # Left crsr
            if text_c[0] == 0:
                if text_c[1] > 0:
                    text_c[0] = mc
                    text_c[1] -= 1
            else:
                text_c[0] -= 1
        elif data == 264:       # Down crsr
            if text_c[1] < mr:
                text_c[1] += 1
        elif data == 265:       # Up crsr
            if text_c[1] > 0:
                text_c[1] -= 1
    dpg.configure_item('cursor',pmin=(text_c[0]*16,text_c[1]*16),pmax=(16+text_c[0]*16,16+text_c[1]*16))

#########################
def get_textures():
    global chars_msx, chars_cbmu, chars_cbml
    global raw_tex, pimg, iimg, raw_screen, raw_tile, raw_icons, raw_icons16, raw_prev, dummy_tex
    pimg = Image.open('assets/splash.gif').convert('RGB')
    cimg = Image.open('assets/charset-msx.png').convert('RGB')
    iimg = Image.open('assets/icons.png').convert('RGB')
    cimg = cimg.resize((cimg.size[0]*2,cimg.size[1]*2),resample=Image.Resampling.NEAREST)
    
    dummy_tex = [0,0,0,0]

    raw_tex = np.asarray(pimg, dtype=np.float32)/255
    raw_chars = np.ascontiguousarray(cimg, dtype=np.float32)/255
    icons = np.ascontiguousarray(iimg, dtype=np.float32)/255

    # tmp = np.zeros((raw_chars.shape[0],raw_chars.shape[1],3),np.float32)
    raw_tile = np.zeros((16,16,3),np.float32)
    raw_prev = np.zeros((32,32,3),np.float32)

    chars_msx = []
    for j in range(0, raw_chars.shape[0], 16):
        for i in range(0, raw_chars.shape[1], 16):
            chars_msx.append(raw_chars[j:j+16, i:i+16])#(np.reshape(raw_chars[j:j+8, i:i+8], (192)))
    chars_msx = np.asarray(chars_msx).astype(np.float32)

    cimg = Image.open('assets/charset-cbm.gif').convert('RGB')
    cimg = cimg.resize((cimg.size[0]*2,cimg.size[1]*2),resample=Image.Resampling.NEAREST)
    raw_chars = np.ascontiguousarray(cimg, dtype=np.float32)/255

    chars_cbmu =[]
    for j in range(0, raw_chars.shape[0]//2, 16):
        for i in range(0, raw_chars.shape[1], 16):
            chars_cbmu.append(raw_chars[j:j+16, i:i+16])#(np.reshape(raw_chars[j:j+8, i:i+8], (192)))
    chars_cbmu = np.asarray(chars_cbmu).astype(np.float32)

    chars_cbml =[]
    for j in range(raw_chars.shape[0]//2, raw_chars.shape[0], 16):
        for i in range(0, raw_chars.shape[1], 16):
            chars_cbml.append(raw_chars[j:j+16, i:i+16])#(np.reshape(raw_chars[j:j+8, i:i+8], (192)))
    chars_cbml = np.asarray(chars_cbml).astype(np.float32)

    for j in range(0, icons.shape[1],32):
        raw_icons.append(np.ascontiguousarray(icons[0:32, j:j+32]))
    for j in range(0, 4):
        raw_icons16.append(np.ascontiguousarray(icons[32:48, j*16:(j*16)+16]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=1, height=1, default_value= dummy_tex, tag='dummy')
        dpg.add_raw_texture(width=320, height=200, default_value= raw_tex, format=dpg.mvFormat_Float_rgb, tag="splash_id")
        dpg.add_raw_texture(width=640, height=400, default_value= raw_screen, format=dpg.mvFormat_Float_rgb, tag="screen")
        dpg.add_raw_texture(width=16, height=16, default_value=raw_tile, format=dpg.mvFormat_Float_rgb, tag='curtile')
        dpg.add_raw_texture(width=32, height=32, default_value=raw_prev, format=dpg.mvFormat_Float_rgb, tag='prevtile')
        dpg.add_raw_texture(width=200, height=240, default_value= prev_tex, format=dpg.mvFormat_Float_rgb, tag="preview_id")
        for i,t in enumerate(chars_msx):
            dpg.add_raw_texture(width=16, height=16, default_value=t, format=dpg.mvFormat_Float_rgb, tag='cmsx'+str(i))
        for i,t in enumerate(chars_cbmu):
            dpg.add_raw_texture(width=16, height=16, default_value=t, format=dpg.mvFormat_Float_rgb, tag='cbmu'+str(i))
        for i,t in enumerate(chars_cbml):
            dpg.add_raw_texture(width=16, height=16, default_value=t, format=dpg.mvFormat_Float_rgb, tag='cbml'+str(i))
        for i,t in enumerate(raw_icons):
            dpg.add_raw_texture(width=32, height=32, default_value=t, format=dpg.mvFormat_Float_rgb, tag='i'+str(i))
        for i,t in enumerate(raw_icons16):
            dpg.add_raw_texture(width=16, height=16, default_value=t, format=dpg.mvFormat_Float_rgb, tag='si'+str(i))
        dpg.add_raw_texture(width=640, height=400, default_value=raw_over, format=dpg.mvFormat_Float_rgba, tag='oimg')

# Check if file exist and prompt for overwrite if so
def check_file(sender,app_data, user_data):
    if file_exists(app_data['file_path_name']):
        dpg.configure_item('owb_id', user_data=[app_data['file_path_name'],user_data])
        dpg.configure_item('ow_id', show=True)
    else:
        save_file(None,None,[app_data['file_path_name'],user_data])

###############################
# Image crop and resize
def frameResize(i_image):
    i_ratio = i_image.size[0] / i_image.size[1]
    in_size = (modes[screen_mode]['geo'][0]*16,modes[screen_mode]['geo'][1]*16)
    dst_ratio = in_size[0]/in_size[1]
    if dst_ratio >= i_ratio:
        i_image = i_image.resize((in_size[0],in_size[0]*i_image.size[1]//i_image.size[0]), Image.LANCZOS)
        box = (0,(i_image.size[1]-in_size[1])/2,i_image.size[0],(i_image.size[1]+in_size[1])/2)
        i_image = i_image.crop(box)
    elif dst_ratio < i_ratio:
        i_image = i_image.resize((in_size[1]*i_image.size[0]//i_image.size[1],in_size[1]),Image.LANCZOS)
        box = ((i_image.size[0]-in_size[0])/2,0,(i_image.size[0]+in_size[0])/2,i_image.size[1])
        i_image = i_image.crop(box)

    return i_image

def open_overlay(sender, app_data):
    global raw_over, over_en
    dpg.hide_item("click_handler")  # Stop mouse click reporting
    try:
        img_over = frameResize(Image.open(app_data['file_path_name']).convert('RGBA'))
        tmp = np.asarray(img_over, dtype=np.float32)/255
        raw_over[0:modes[screen_mode]['geo'][1]*16,0:modes[screen_mode]['geo'][0]*16] = tmp[:]
        raw_over[:,:,3] = .25
        dpg.configure_item('overlay', show=True)
        dpg.configure_item('over_b', tint_color=(255,255,255))
        over_en = True
    except:
        pass

def sync_matrix():
    global raw_screen, raw_tile
    chars = chars_msx if screen_mode == 'MSX Screen 2 - RetroTerm' else chars_cbmu if charset == 'Upper/GFX' else chars_cbml
    # for i,cell in enumerate(matrix):
    mc = modes[screen_mode]['geo'][0]
    mr = modes[screen_mode]['geo'][1]
    for i in range(mr):
        for j in range(mc):
            x = (j)*16
            y = (i)*16
            cell = matrix[j,i]
            raw_tile[np.where((chars[cell[0]]==[0,0,0]).all(axis=2))] = np.array(palette[cell[1]],dtype=np.float32)/255
            raw_tile[np.where((chars[cell[0]]==[1,1,1]).all(axis=2))] = np.array(palette[cell[2]],dtype=np.float32)/255
            raw_screen[y:y+16, x:x+16] = raw_tile
    raw_tile[np.where((chars[curchar]==[0,0,0]).all(axis=2))] = np.array(palette[colors[0]],dtype=np.float32)/255
    raw_tile[np.where((chars[curchar]==[1,1,1]).all(axis=2))] = np.array(palette[colors[1]],dtype=np.float32)/255

# Save file
def save_file(sender, app_data, user_data):
    dpg.configure_item("ow_id", show=False)
    dpg.configure_item('save',show=True,callback=lambda: save_file(sender,app_data,user_data))
    save_seq(user_data[0])

def new_work():
    clear_screen()
    undobuffer.clear()
    dpg.configure_item('save',show=False)


def open_seq(sender, app_data, user_data):
    global background,colors, matrix, undobuffer, redobuffer
    filename = app_data['file_path_name']
    with open(filename,'rb') as fo:
        c = 0
        r = 0
        mc = modes[screen_mode]['geo'][0]
        mr = modes[screen_mode]['geo'][1]
        cmd = False
        rvs = False
        while True:
            char = fo.read(1)
            if not char:
                break
            char = char[0]
            if screen_mode != 'Commodore 64':
                if char == 1:   # Colors/Extended gfx
                    char = fo.read(1)
                    if not char:
                        break
                    char = char[0]
                    if char < 16:
                        colors[1] = char-1
                        dpg.set_value('color1', palette[colors[1]])
                    elif char < 32:
                        colors[0] = char-17
                        dpg.set_value('color0', palette[colors[0]])
                    elif char in range(0x40,0x60):
                        matrix[c,r]=[char-0x40,colors[0],colors[1]]
                        c += 1
                        if c > mc-1:
                            c = 0
                            r = r+1 if r < mr-1 else mr-1
                    elif char in (0x60,0x61):
                        matrix[c,r]=[char+0x9e,colors[0],colors[1]]
                        c += 1
                        if c > mc-1:
                            c = 0
                            r = r+1 if r < mr-1 else mr-1
                elif char == 0x0c:  #CLS
                    c = 0
                    r = 0
                    background = colors[0]
                    dpg.set_value('color2', palette[colors[0]])
                    matrix[:] = [32,colors[0],colors[1]]
                elif char == 0x0d: #CR
                    c = 0
                    r = r+1 if r < mr-1 else mr-1
                elif char == 0x12:  # Insert
                    matrix[:,r] = np.insert(matrix[:,r],(c*3),[32,colors[0],colors[1]]).reshape([mc+1,3])[:-1]
                elif char == 0x1d:  # CRSR left
                    c -= 1
                    if c < 0:
                        c = 0
                        r = r-1 if r > 0 else 0
                elif char >= 32:    #Chars
                    matrix[c,r]=[char,colors[0],colors[1]]
                    c += 1
                    if c > mc-1:
                        c = 0
                        r = r+1 if r < mr-1 else r
            else:
                if char == 255:    # CMD_ON
                    cmd = True
                if char == 254:    # CMD_OFF
                    cmd = False
                if cmd:
                    if char == 0x90:
                        char = fo.read(3)
                        background = char[2]
                        colors[0] = background
                        dpg.set_value('color0', palette[colors[0]])
                        dpg.set_value('color2', palette[background])
                        dpg.set_value('color3', palette[char[1]])
                else:
                    if char.to_bytes(1,'big') in petcolors:    #Colors
                        colors[1] = petcolors.index(char.to_bytes(1,'big'))
                        dpg.set_value('color1', palette[colors[1]])
                    elif char == 0x93:  #Clear
                        c = 0
                        r = 0
                        matrix[:] = [32,colors[0],colors[1]]
                    elif char in [0x0d,0x8d]:  #CR
                        c = 0
                        r = r+1 if r < mr-1 else mr-1
                        rvs = False
                    elif char == 0x94:  #Insert
                        matrix[:,r] = np.insert(matrix[:,r],(c*3),[32,colors[0],colors[1]]).reshape([41,3])[:-1]
                    elif char == 0x9d:  #Csrs left
                        c -= 1
                        if c < 0:
                            c = 0
                            r = r-1 if r > 0 else 0
                    elif char == 0x92:  #RVS off
                        rvs = False
                    elif char == 0x12:  #RVS on
                        rvs = True
                    elif char == 0x8e:  #Upper/GFX
                        set_charset('','Upper/GFX')
                        dpg.configure_item('charset',default_value='Upper/GFX')
                    elif char == 0x0e:  #Lower/Upper
                        set_charset('','Lower/Upper')
                        dpg.configure_item('charset',default_value='Lower/Upper')
                    else:
                        if not rvs:
                            if char in range(32,64):                        # -> 32-63
                                matrix[c,r]=[char,colors[0],colors[1]]
                            elif char in range(64,96):                      # -> 0-31
                                matrix[c,r]=[char-64,colors[0],colors[1]]
                            elif char in range(96,128):                     # -> 64-95
                                matrix[c,r]=[char-32,colors[0],colors[1]]
                            elif char in range(160,192):                    # -> 96-127
                                matrix[c,r]=[char-64,colors[0],colors[1]]
                            elif char in range(192,256):                    # -> 64-127
                                matrix[c,r]=[char-128,colors[0],colors[1]]
                        else:
                            if char in range(32,64):                        # -> 160-191
                                matrix[c,r]=[char+128,colors[0],colors[1]]
                            elif char in range(64,96):                      # -> 128-159
                                matrix[c,r]=[char+64,colors[0],colors[1]]
                            elif char in range(96,128):                     # -> 192-223
                                matrix[c,r]=[char+96,colors[0],colors[1]]
                            elif char in range(160,192):                    # -> 224-255
                                matrix[c,r]=[char+64,colors[0],colors[1]]
                            elif char in range(192,256):                    # -> 192-255
                                matrix[c,r]=[char,colors[0],colors[1]]
                        c += 1
                        if c > mc-1:
                            c = 0
                            r = r+1 if r < mr-1 else r


        sync_matrix()
        undobuffer.clear()
        redobuffer.clear()
        dpg.configure_item('save',show=True,callback=lambda: save_file(sender,app_data,[filename]))

        
def save_seq(filename):

    def scr2pet(c):
        nonlocal rvs
        bin = b''
        if c < 128:
            if rvs:
                rvs = False
                bin += b'\x92'  #RVS off
            # Normal
            if c in range(32):
                bin += int(c+64).to_bytes(1,'big')  # -> 64-95
            elif c in range(32,64):
                bin += int(c).to_bytes(1,'big')     # -> 32-63
            elif c in range(64,96):
                bin+= int(c+32).to_bytes(1,'big')   # -> 96-127
            elif c in range(96,128):
                bin+= int(c+64).to_bytes(1,'big')   # -> 160-191
        else:
            if not rvs:
                rvs = True
                bin += b'\x12'  #RVS on
            # Reversed
            if c in range(128,160):
                bin += int(c-64).to_bytes(1,'big')
            elif c in range(160,192):
                bin += int(c-128).to_bytes(1,'big')
            elif c in range(192,224):
                bin+= int(c-96).to_bytes(1,'big')
            elif c in range(224,256):
                bin+= int(c-64).to_bytes(1,'big')
        return bin



    output = deque()

    if screen_mode == 'Commodore 64':
        bin = b'\xff\x90\x00'+border.to_bytes(1,'big')+background.to_bytes(1,'big')+b'\xfe\x93'+b'\x8e' if charset == 'Upper/GFX' else b'\x0e'  # Set Text mode, black border, background color, clear screen, set charset
        current = [modes[screen_mode]['default'][0],background,-1]
        empty = [modes[screen_mode]['default'][0],background]
    else:
        bg = background+1+16
        bin = b'\x01'+bg.to_bytes(1,'big')+b'\x0c'  # Set paper color, clear screen
        current = [modes[screen_mode]['default'][0],bg-16,-1]
        empty = [modes[screen_mode]['default'][0],bg-16]
    default = modes[screen_mode]['default'][0]
    rvs = False
    output.append(bin)
    bin = b''
    mc = modes[screen_mode]['geo'][0]
    mr = modes[screen_mode]['geo'][1]
    for r in range(mr):
        buffer = b''        # buffer of empty spaces
        for col in range(mc):
            bin = b''
            cell = matrix[col,r].copy()
            if screen_mode != 'Commodore 64':
                cell[1:] += 1
            if not np.array_equal(cell,current):
                #bin += buffer
                if buffer != b'':
                    output.append(buffer)
                    buffer = b''
                c,b,f = cell    # character, bg, fg
                # if np.array_equal(np.flip(cell[1:],0),current[1:]): # Inverse
                #     bin += b'\x1a' if rvs else b'\x19'
                #     rvs = not rvs
                # else:
                if screen_mode != 'Commodore 64':
                    if f != current[2]:     #FG color
                        bin += b'\x01'+int(f+(16*rvs)).to_bytes(1,'big')
                    if b != current[1]:     #BG color
                        bin += b'\x01'+int(b+(16*(not rvs))).to_bytes(1,'big')
                    if not np.array_equal(cell[:2],empty):
                        if c < 0x20:
                            bin += b'\x01'+int(c+0x40).to_bytes(1,'big')
                        elif c in (254,255):
                            bin += b'\x01'+int(c-0x9e).to_bytes(1,'big')
                        else:
                            bin += int(c).to_bytes(1,'big')
                    else:
                        buffer += b' '
                else:
                    if f != current[2]:     #FG color
                        bin += petcolors[f]
                    if c != default or rvs:
                        bin += scr2pet(c)
                    else:
                        buffer += int(c).to_bytes(1,'big')
                output.append(bin)
                current = [c,b,f]
                if col == mc-1 and r == mr-1 and buffer == b'':   # Character on the last screen cell?
                    output.pop()    # Discard last cell
                    penult = output.pop()
                    if len(penult) > 1 and len(set(penult)) == 1 and penult[0] != 1:
                        output.append(penult[:-1])  #2nd to last element were empty n-spaces, reinsert n-spaces-1
                    c,b,f = matrix[col,r]   # Last character
                    if screen_mode == 'Commodore 64':
                        rvs = c < 128
                        bin = petcolors[f]+scr2pet(c)
                    else:
                        bin = b'\x01'+int(f+1+(16*rvs)).to_bytes(1,'big')+b'\x01'+int(b+1+(16*(not rvs))).to_bytes(1,'big')
                        if c < 0x20:
                            bin += b'\x01'+int(c+0x40).to_bytes(1,'big')
                        elif c in (254,255):
                            bin += b'\x01'+int(c-0x9e).to_bytes(1,'big')
                        else:
                            bin += int(c).to_bytes(1,'big')
                    if screen_mode == 'Commodore 64':   # left crsr + insert
                        bin += b'\x9d\x94'
                    else:
                        bin += b'\x1d\x12'
                    c,b,f = matrix[mc-2,mr-1]   # 2nd to last character
                    if screen_mode == 'Commodore 64':
                        rvs = c < 128
                        bin += petcolors[f]+scr2pet(c)
                    else:
                        bin += b'\x01'+int(f+1+(16*rvs)).to_bytes(1,'big')+b'\x01'+int(b+1+(16*(not rvs))).to_bytes(1,'big')
                        if c < 0x20:
                            bin += b'\x01'+int(c+0x40).to_bytes(1,'big')
                        elif c in (254,255):
                            bin += b'\x01'+int(c-0x9e).to_bytes(1,'big')
                        else:
                            bin += int(c).to_bytes(1,'big')
                    output.append(bin)
            elif np.array_equal(cell[:2],empty):
                buffer += b' '
                rvs = False
            else:
                output.append(buffer)
                if screen_mode != 'Commodore 64':
                    if cell[0] < 0x20:
                        output.append(b'\x01'+int(cell[0]+0x40).to_bytes(1,'big'))
                    elif cell[0] in (254,255):
                        output.append(b'\x01'+int(cell[0]-0x9e).to_bytes(1,'big'))
                    else:
                        output.append(int(cell[0]).to_bytes(1,'big'))
                else:
                    output.append(scr2pet(c))
                buffer = b''
        if buffer != b'' and r < mr:
            output.append(b'\x0d')
            rvs = False
    with open(filename,'wb') as file:
        while len(output)>0:
            file.write(output.popleft())
                    

###########################################################3
# GUI elements only after this call
dpg.create_context()
dpg.create_viewport(title='RetroBBS Screen Editor', width=1024,height=550, resizable=False)

dpg.setup_dearpygui()

get_textures()

with dpg.window(label="Quit Program", modal=True, show=False, id="quit_id", no_title_bar=True, pos=(200,200), no_resize=True):
    dpg.add_text("Exit program?")
    dpg.add_separator()
    with dpg.group(horizontal=True, pos=(0,60)):
        dpg.add_button(label="OK", width=75, callback=quit_callback)
        dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("quit_id", show=False))

with dpg.window(label="About", modal=True, show=False, id="aboutw_id", no_title_bar=True,pos=(200,200),no_resize=False):
    dpg.add_text('RetroBBS Screen Editor')
    dpg.add_image_button(texture_tag="splash_id", frame_padding=0, callback=about_callback)
    dpg.add_text('Code: Pablo Roldán (Durandal)')
    dpg.add_text('Retroterm MSX Font: Jorge Castillo (Pastbytes)')
    dpg.add_text('©2024 Retrocomputacion')

# Open file dialog
with dpg.file_dialog(label="Open overlay", directory_selector=False, show=False, callback=open_overlay, tag="open_dialog", min_size=(700,400)):
    dpg.add_file_extension("", color=(255, 155, 50, 255))
    dpg.add_file_extension("Image files{.gif,.GIF,.jpg,.JPG,.jpeg,.JPEG,.png,.PNG}", color=(0, 255, 64, 255))
    with dpg.child_window(width= 240, height=290):
        dpg.add_text("Preview")
        dpg.add_image(texture_tag="preview_id", width=200, height=240)
    with dpg.handler_registry(show=False, tag='click_handler'):
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=fileclick)

# Overwrite dialog
with dpg.window(label="File Exists", modal=True, show=False, id="ow_id", no_title_bar=True, pos=(200,200), no_resize=True):
    dpg.add_text("Overwrite file?")
    dpg.add_separator()
    with dpg.group(horizontal=True, pos=(0,60)):
        dpg.add_button(label="OK", width=75, tag="owb_id", callback=save_file)
        dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("ow_id", show=False))

with dpg.window(tag="MainW", no_scrollbar= True):
    with dpg.item_handler_registry(tag="drag_handler", show=True):
        dpg.add_item_hover_handler(callback=drag_handler)
        dpg.add_item_focus_handler(callback=drag_handler)
        dpg.add_item_deactivated_handler(callback=release_handler)
        dpg.add_item_activated_handler(callback=click_handler)

    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label='New', callback=new_work)
            dpg.add_menu_item(label='Open', callback=show_open)
            dpg.add_menu_item(label="Save", tag='save', show=False)
            dpg.add_menu_item(label="Save As...", callback=show_save, user_data='s_seq')
            dpg.add_separator()
            dpg.add_menu_item(label='Open overlay...', callback=show_dialog)
            dpg.add_menu_item(label="Quit", callback= lambda: dpg.configure_item("quit_id", show = True))
        with dpg.menu(label="Mode"):
            dpg.add_radio_button(('MSX Screen 2 - RetroTerm','Commodore 64'), default_value='MSX',tag='s_mode', callback=set_screenmode)
        with dpg.menu(label="Help"):
            # dpg.add_menu_item(label="Help", callback=print_me)
            dpg.add_menu_item(label="About...", callback= show_about)
    with dpg.group(horizontal=True, horizontal_spacing=0):

        with dpg.handler_registry(tag="key_handler", show=True):
            dpg.add_key_press_handler(key=dpg.mvKey_Z, callback=short_handler)  #Undo
            dpg.add_key_press_handler(key=dpg.mvKey_Y, callback=short_handler)  #Redo
            dpg.add_key_press_handler(key=dpg.mvKey_G, callback=short_handler)  #Grid
            dpg.add_key_press_handler(key=dpg.mvKey_O, callback=short_handler)  #Overlay
            dpg.add_key_press_handler(key=dpg.mvKey_Home, callback=short_handler)  #Clear

        with dpg.handler_registry(tag="text_handler", show=False):
            dpg.add_key_press_handler(callback=text_mode)           #Text mode

        with dpg.group(horizontal_spacing=0, pos=(30,50)):
            with dpg.drawlist(width=512, height=384, tag='mainimg', callback= flood_fill):
                dpg.draw_image('screen', (0,0), (640,400), uv_min=(0,0), uv_max=(1,1))
                dpg.draw_image('oimg', (0,0), (640,400), uv_min=(0,0), uv_max=(1,1), tag='overlay', show=False)
                with dpg.draw_layer(label='grid', tag='grid'):
                    for x in range(0,640,16):
                        dpg.draw_line((x,0),(x,400), color=(128,128,128,64), thickness=1)
                    for y in range(0,400,16):
                        dpg.draw_line((0,y),(640,y), color=(128,128,128,64), thickness=1)
                with dpg.draw_layer(show=True):
                    dpg.draw_rectangle(color=(255,0,0), thickness=2, pmin=(0,0), pmax=(15,15), tag='cursor', show=False)
                    dpg.draw_rectangle(color=(0,255,0), thickness=1, pmin=(0,0), pmax=(0,0), tag='selection', show=False)
            dpg.bind_item_handler_registry("mainimg", "drag_handler")
            with dpg.table(header_row=False, width=520):
                dpg.add_table_column()
                dpg.add_table_column()
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_color_button(default_value=(0,0,0), width=10, height=16, label="Background color", tag='color2', callback=color_selector, no_alpha=True)
                        dpg.add_text('Background')
                    with dpg.group(horizontal=True):
                        dpg.add_color_button(default_value=(0,0,0), width=10, height=16, label="Border color", tag='color3', callback=color_selector, no_alpha=True)
                        dpg.add_text('Border')
                    # dpg.add_spacer()
                    dpg.add_text('X:0 Y:0', tag='coords')
                    with dpg.group(horizontal=True):
                        dpg.add_text(f'0x{modes[screen_mode]["default"][0]}', tag='char_hover')
                        dpg.add_color_button(default_value=(255,255,255), label='cell foreground color', width=10, height=16, tag='fgcolor', no_alpha=True)
                        dpg.add_color_button(default_value=(0,0,0), label='cell background color', width=10, height=16, tag='bgcolor', no_alpha=True)
            toolbar = [['i0',clear_screen,'clear_b',(255,255,255),'Clear canvas'],
                       ['i1',undoredo,'undo',(255,255,255),'Undo'],
                       ['i2',undoredo,'redo',(255,255,255),'Redo'],
                       ['i4',grid,'grid_b',(255,255,255),'Show grid'],
                       ['i5',overlay,'over_b',(128,128,128),'Show overlay image'],
                       ['i3',set_mode,'draw_b',(255,255,255),'Draw'],
                       ['i8',set_mode,'paint_b',(128,128,128),'Recolor'],
                       ['i6',set_mode,'fill_b',(128,128,128),'Flood fill'],
                       ['i7',set_mode,'text_b',(128,128,128),'Text'],
                       ['i9',set_mode,'brush_b',(128,128,128),'Brush'],
                       ['i10',set_mode,'select_b',(128,128,128),'Select']]
            with dpg.group(horizontal=True, horizontal_spacing=0):
                for tb in toolbar:
                    dpg.add_image_button(texture_tag=tb[0], width=32, height=32, callback=tb[1], tag=tb[2], tint_color=tb[3])
                    with dpg.tooltip(tb[2],delay=0.25):
                        dpg.add_text(tb[4])
            with dpg.group(horizontal=True, horizontal_spacing=0, indent=360, show=False, tag='flip'):
                dpg.add_image_button(texture_tag='si1', width=16, height=16, tag='flip_x', callback=flip, frame_padding=2, show=True)
                dpg.add_image_button(texture_tag='si2', width=16, height=16, tag='flip_y', callback=flip, frame_padding=2, show=True)
                with dpg.tooltip('flip_x',delay=0.25):
                    dpg.add_text('Flip-X')
                with dpg.tooltip('flip_y',delay=0.25):
                    dpg.add_text('Flip-Y')

        with dpg.group():
            with dpg.group(tag='msx_chars', show=True):
                for j in range(0,16):
                    with dpg.group(horizontal=True, horizontal_spacing=0, pos=(700,50+(17*j))):
                        for i in range(0,16):
                            if ((j*16)+i) in(0,):    #254,255):
                                dpg.add_image_button(texture_tag='dummy', width=16, height=16, indent=-1, frame_padding=1,
                                                    tint_color=(128,128,128))
                            else:
                                dpg.add_image_button(texture_tag='cmsx'+str((j*16)+i), width=16, height=16, indent=-1, tag='b'+str((j*16)+i),frame_padding=1,
                                                    callback=select_char, tint_color=(128,128,128))
            with dpg.group(tag='cbmu_chars', show=False):
                for j in range(0,16):
                    with dpg.group(horizontal=True, horizontal_spacing=0, pos=(700,50+(17*j))):
                        for i in range(0,16):
                                dpg.add_image_button(texture_tag='cbmu'+str((j*16)+i), width=16, height=16, indent=-1, tag='d'+str((j*16)+i),frame_padding=1,
                                                    callback=select_char, tint_color=(128,128,128))
            with dpg.group(tag='cbml_chars', show=False):
                for j in range(0,16):
                    with dpg.group(horizontal=True, horizontal_spacing=0, pos=(700,50+(17*j))):
                        for i in range(0,16):
                                dpg.add_image_button(texture_tag='cbml'+str((j*16)+i), width=16, height=16, indent=-1, tag='e'+str((j*16)+i),frame_padding=1,
                                                    callback=select_char, tint_color=(128,128,128))
            dpg.add_image('prevtile', width=32, height=32, pos=(800,335))
            dpg.add_combo(['Upper/GFX','Lower/Upper'], label='Charset',default_value='Upper/GFX',tag='charset',pos=(850,350),width=100,show=False, callback=set_charset)
            dpg.add_text('0x20',tag='prevcode', pos=(802,369))
            dpg.add_color_button(palette[colors[0]], tag='color0', pos=(725,351), width=32, height=32, no_alpha=True)
            dpg.add_color_button(palette[colors[1]], tag='color1', pos=(741,335), width=32, height=32, no_alpha=True)
            dpg.add_image_button(texture_tag='si0', width=16, height=16, callback=swap_colors, pos=(700,345), frame_padding=2, tag='swap')
            with dpg.tooltip('swap',delay=1):
                dpg.add_text('Swap')
            c_buttons = []
            with dpg.group(horizontal=True, pos=(700,400), horizontal_spacing=0, tag='msxcolors'):
                for i,c in enumerate(modes['MSX Screen 2 - RetroTerm']['palette']):
                    c_buttons.append(dpg.add_color_button(c, tag='bcolor'+str(i), height=32, no_alpha=True))
            with dpg.group(horizontal=True, pos=(700,400), horizontal_spacing=0, tag='c64colors', show=False):
                for i,c in enumerate(modes['Commodore 64']['palette']):
                    c_buttons.append(dpg.add_color_button(c, tag='ccolor'+str(i), width=18, height=32, no_alpha=True))
            with dpg.item_handler_registry(tag='phandler'):
                dpg.add_item_clicked_handler(0, callback=set_color)
                dpg.add_item_clicked_handler(1, callback=set_color)
            for cb in c_buttons:
                dpg.bind_item_handler_registry(cb,'phandler')
 
dpg.show_viewport()
dpg.set_primary_window("MainW",True)

while dpg.is_dearpygui_running() and not Quit:
    dpg.render_dearpygui_frame()

dpg.destroy_context()
