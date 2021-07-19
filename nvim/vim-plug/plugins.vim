" auto-install vim-plug
if empty(glob('~/.config/nvim/autoload/plug.vim'))
  silent !proxychains4 curl -fLo ~/.config/nvim/autoload/plug.vim --create-dirs
    \ https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
  "autocmd VimEnter * PlugInstall
  "autocmd VimEnter * PlugInstall | source $MYVIMRC
endif

call plug#begin('~/.config/nvim/autoload/plugged')

    " Better Syntax Support
    Plug 'sheerun/vim-polyglot'
    " File Explorer
    " Plug 'scrooloose/NERDTree'
    " Easy Motion
    Plug 'easymotion/vim-easymotion'
    " Auto pairs for '(' '[' '{'
    Plug 'jiangmiao/auto-pairs'
    " Onedark Theme
    Plug 'joshdick/onedark.vim'
    " Stable version of coc
    Plug 'neoclide/coc.nvim', {'branch': 'release'}
    " Keeping up to date with master
    Plug 'neoclide/coc.nvim', {'do': 'yarn install --frozen-lockfile'}
    " Install airline themes
    Plug 'vim-airline/vim-airline'
    Plug 'vim-airline/vim-airline-themes'
    " Ranger Plugin 
    " Plug 'kevinhwang91/rnvimr', {'do': 'make sync'}
    " Comment Plugin
    Plug 'tpope/vim-commentary'
    " FZF & vim-rooter
    Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
    Plug 'junegunn/fzf.vim'
    Plug 'airblade/vim-rooter'
    " Start Page Customize
    Plug 'mhinz/vim-startify'
    " Which Key Helper
    Plug 'liuchengxu/vim-which-key'
    " Add Icon to File
    Plug 'ryanoasis/vim-devicons'
    " Surround vim
    Plug 'kana/vim-surround'

call plug#end()
