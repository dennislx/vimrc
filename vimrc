source $VIMRUNTIME/vimrc_example.vim
source $VIMRUNTIME/mswin.vim

set diffexpr=MyDiff()
function MyDiff()
  let opt = '-a --binary '
  if &diffopt =~ 'icase' | let opt = opt . '-i ' | endif
  if &diffopt =~ 'iwhite' | let opt = opt . '-b ' | endif
  let arg1 = v:fname_in
  if arg1 =~ ' ' | let arg1 = '"' . arg1 . '"' | endif
  let arg1 = substitute(arg1, '!', '\!', 'g')
  let arg2 = v:fname_new
  if arg2 =~ ' ' | let arg2 = '"' . arg2 . '"' | endif
  let arg2 = substitute(arg2, '!', '\!', 'g')
  let arg3 = v:fname_out
  if arg3 =~ ' ' | let arg3 = '"' . arg3 . '"' | endif
  let arg3 = substitute(arg3, '!', '\!', 'g')
  if $VIMRUNTIME =~ ' '
    if &sh =~ '\<cmd'
      if empty(&shellxquote)
        let l:shxq_sav = ''
        set shellxquote&
      endif
      let cmd = '"' . $VIMRUNTIME . '\diff"'
    else
      let cmd = substitute($VIMRUNTIME, ' ', '" ', '') . '\diff"'
    endif
  else
    let cmd = $VIMRUNTIME . '\diff'
  endif
  let cmd = substitute(cmd, '!', '\!', 'g')
  silent execute '!' . cmd . ' ' . opt . arg1 . ' ' . arg2 . ' > ' . arg3
  if exists('l:shxq_sav')
    let &shellxquote=l:shxq_sav
  endif
endfunction

set hlsearch
set nocompatible
set nobackup
set nowritebackup
set noswapfile
set ruler
set wrap
set laststatus=2
set wmnu wildmode=longest:full
set expandtab tabstop=4
set si ai ci cinkeys-=0# cinoptions=g0,:0
set shiftwidth=4
set softtabstop=4
set lcs=eol:$,tab:\|\
set backspace=indent,eol,start
set fileencodings=utf-8,cp936
"set relativenumber

" Plugin 设置
call plug#begin('$VIM/vimfiles/bundle/')
Plug 'iamcco/markdown-preview.nvim', { 'do': 'cd app & yarn install'  }
" Plug 'iamcco/markdown-preview.nvim', { 'do': { -> mkdp#util#install() }, 'for': ['markdown', 'vim-plug']}
call plug#end()

" Vundle 设置
" set the runtime path to include Vundle and initialize
set rtp+=$VIM/vimfiles/bundle/Vundle.vim/
call vundle#begin('$VIM/vimfiles/bundle/')
" alternatively, pass a path where Vundle should install plugins
" call vundle#begin('~/some/path/here')
"
" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
"
" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" " plugin on GitHub repo
" "Plugin 'mattn/emmet-vim'
" "Plugin 'landonb/dubs_html_entities'
Plugin 'tiagofumo/vim-nerdtree-syntax-highlight'
Plugin 'sheerun/vim-polyglot'
" Plugin 'tpope/vim-surround'
" Plugin 'Raimondi/delimitMate'
" Plugin 'vim-syntastic/syntastic'
" Plugin 'ryanoasis/vim-devicons'
Plugin 'rafi/awesome-vim-colorschemes'
" Plugin 'drewtempelmeyer/palenight.vim'
" Plugin 'ervandew/supertab'
" Plugin 'lilydjwg/colorizer'
" Plugin 'kshenoy/vim-signature'
Plugin 'MarcWeber/vim-addon-mw-utils'
Plugin 'tomtom/tlib_vim'
" Plugin 'easymotion/vim-easymotion'
Plugin 'vim-airline/vim-airline'
Plugin 'vim-airline/vim-airline-themes'
Plugin 'ctrlpvim/ctrlp.vim'
Plugin 'scrooloose/nerdcommenter'
Plugin 'jlanzarotta/bufexplorer'
Plugin 'arcticicestudio/nord-vim'
" plugin from http://vim-scripts.org/vim/scripts.html
"Plugin 'L9'
" Git plugin not hosted on GitHub
"Plugin 'https://github.com/wincent/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
"Plugin 'file:///home/gmarik/path/to/plugin'
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
"Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" Install L9 and avoid a Naming conflict if you've already installed a
" different version somewhere else.
"Plugin 'ascenator/L9', {'name': 'newL9'}

" All of your Plugins must be added before the following line

call vundle#end()            

" @airline
" let g:airline#extensions#tabline#enabled = 1   是否打开tabline
"这个是安装字体后 必须设置此项" 
let g:airline_powerline_fonts = 1
set laststatus=2  "永远显示状态栏
let g:airline_theme='bubblegum' "选择主题

" set t_Co=256      "在windows中用xshell连接打开vim可以显示色彩
colorscheme nord
" let g:nord_cursor_line_number_background = 1
" let g:nord_uniform_diff_background = 1
syntax enable
hi Visual cterm=bold ctermfg=122 ctermbg=233

" document tree
Bundle 'scrooloose/nerdtree'
Bundle 'Auto-Pairs'
" Nerdtree快捷键设置
let NERDTreeWinPos='right'
let NERDTreeWinSize=30
map <F2> :NERDTreeToggle<CR>

" ctrlp快速打开文件
let g:ctrlp_map = '<c-p>'
let g:ctrlp_cmd = 'CtrlP'
set wildignore+=*\\node_modules\\*,*.git*,*.svn*,*.zip*,*.exe* " 使用vim的忽略文件

let g:NERDSpaceDelims = 1
nnoremap <silent><F8> :BufExplorer<CR>

" Enable folding
" set foldmethod=indent
" set foldlevel=99
" "Enable folding with the spacebar
" nnoremap <space> za

" au BufNewFile,BufRead *.py,*.java,*.cpp,*.c,*.rkt,*.h
    " \ set tabstop=4 |
    " \ set softtabstop=4 |
    " \ set shiftwidth=4 |
    " \ set textwidth=120 |
    " \ set expandtab |
    " \ set autoindent |
    " \ set fileformat=unix | 

" To ignore plugin indent changes, instead use:
filetype plugin on
filetype plugin indent on

" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just
" :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal

" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line

" vim-signature 
" hi SignColumn ctermbg=NONE guibg=#131313
" hi SignatureMarkText ctermbg=NONE guibg=#131313 gui=bold term=bold cterm=bold

" 重新定义快捷键
" imap <M-space> <Plug>snipMateNextOrTrigger
" smap <M-space> <Plug>snipMateNextOrTrigger
" xmap <M-space> <Plug>snipMateVisual

"取消highlight在搜索后
nnoremap <esc> :noh<return>
"定义leader键位
let mapleader = ","
let g:mapleader = ","
"comment快捷键
nmap <C-_> <Plug>NERDCommenterToggle
vmap <C-_> <Plug>NERDCommenterToggle<CR>gv
"在normal模式下光标在不同窗口间的移动
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l
"看到段落太紧，给下/上留空白
nnoremap <silent> zj o<Esc>k
nnoremap <silent> zk O<Esc>j
"在insert模式下移动光标
"inoremap <M-h> <left>
inoremap <C-j> <down>
inoremap <C-k> <up>
inoremap <C-l> <right>
"预览Markdown文件
map <F5> <Plug>MarkdownPreview
" map <leader>q <Plug>MarkdownPreviewStop

"不明白这个是什么
autocmd GUIEnter * simalt ~x

set nu!
set encoding=utf-8
set termencoding=utf-8
set fileencoding=utf-8
set fileencodings=ucs-bom,utf-8,chinese,cp936

" run code
" nnoremap \ :te<enter>

" autocmd filetype python nnoremap <f5> :w <bar> :!python3 % <cr>
" autocmd filetype cpp nnoremap <f5> :w <bar> !g++ -std=c++11 % -o %:r && ./%:r <cr>
" autocmd filetype c nnoremap <f5> :w <bar> !make %:r && ./%:r <cr>
" autocmd filetype java nnoremap <f5> :w <bar> !javac % && java %:r <cr>

" 文件查找(容易死机，赶紧攒钱换台电脑装Linux)
" set path+=** 
" set wildmenu
