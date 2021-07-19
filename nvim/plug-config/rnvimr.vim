let g:rnvimr_ex_enable = 1

tnoremap <silent> <M-i> <C-\><C-n>:RnvimrResize<CR>
nnoremap <silent> <M-o> :RnvimrToggle<CR>
tnoremap <silent> <M-o> <C-\><C-n>:RnvimrToggle<CR>

let g:rnvimr_layout = {
            \ 'relative': 'editor',
            \ 'width': float2nr(round(0.85 * &columns)),
            \ 'height': float2nr(round(0.85 * &lines)),
            \ 'col': float2nr(round(0.1 * &columns)),
            \ 'row': float2nr(round(0.1 * &lines)),
            \ 'style': 'minimal'
            \ }

