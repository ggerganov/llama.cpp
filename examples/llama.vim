" sample config:
"
"   - Ctrl+F - trigger FIM completion
"
" copy paste this in your .vimrc:
"
"augroup llama_cpp
"    autocmd!
"    autocmd InsertEnter * inoremap <buffer> <silent> <C-F> <Esc>:call llama#fim()<CR>a
"augroup END
"

" color of the suggested text
highlight llama_hl_hint guifg=#ff772f
highlight llama_hl_info guifg=#77ff2f

let s:default_config = {
    \ 'endpoint':    'http://127.0.0.1:8012/infill',
    \ 'n_prefix':    128,
    \ 'n_suffix':    128,
    \ 'n_predict':   64,
    \ 'n_probs':     3,
    \ 'temperature': 0.1,
    \ 'stop':        ["\n"]
    \ }

let g:llama_config = get(g:, 'llama_config', s:default_config)

function! llama#fim() abort
    let l:t_start = reltime()

    let l:pos_x = col('.')
    let l:pos_y = line('.')
    let l:max_y = line('$')

    let l:lines_prefix = getline(max([1, l:pos_y - g:llama_config.n_prefix]), l:pos_y - 1)
    let l:lines_suffix = getline(l:pos_y + 1, min([l:max_y, l:pos_y + g:llama_config.n_suffix]))

    let l:line_cur        = getline('.')
    let l:line_cur_prefix = strpart(l:line_cur, 0, l:pos_x)
    let l:line_cur_suffix = strpart(l:line_cur, l:pos_x)

    let l:prefix = ""
        \ . join(l:lines_prefix, "\n")
        \ . "\n"
        \ . l:line_cur_prefix

    let l:suffix = ""
        \ . l:line_cur_suffix
        \ . "\n"
        \ . join(l:lines_suffix, "\n")
        \ . "\n"

    let l:request = json_encode({
        \ 'prompt':         "",
        \ 'input_prefix':   l:prefix,
        \ 'input_suffix':   l:suffix,
       "\ 'stop':           g:llama_config.stop,
        \ 'n_predict':      g:llama_config.n_predict,
       "\ 'n_probs':        g:llama_config.n_probs,
        \ 'penalty_last_n': 0,
        \ 'temperature':    g:llama_config.temperature,
        \ 'top_k':          5,
        \ 'infill_p':       0.20,
        \ 'infill_p_eog':   0.001,
        \ 'stream':         v:false,
        \ 'samplers':       ["top_k", "infill"]
        \ })

    " request completion from the server
    let l:curl_command = printf(
        \ "curl --silent --no-buffer --request POST --url %s --header \"Content-Type: application/json\" --data %s",
        \ g:llama_config.endpoint, shellescape(l:request)
        \ )

    let l:can_accept = v:true
    let l:has_timing = v:false

    let l:n_prompt    = 0
    let l:t_prompt_ms = 1.0
    let l:s_prompt    = 0

    let l:n_gen    = 0
    let l:t_gen_ms = 1.0
    let l:s_gen    = 0

    let s:content = []

    let l:raw = system(l:curl_command)
    if l:can_accept && v:shell_error
        call add(s:content, "<| curl error: is the server on? |>")
        let l:can_accept = v:false
    endif

    if l:can_accept && l:raw == ""
        call add(s:content, "<| empty response: is the server on? |>")
        let l:can_accept = v:false
    endif

    " get the generated suggestion
    if l:can_accept
        let l:response = json_decode(l:raw)

        for l:part in split(get(l:response, 'content', ''), "\n", 1)
            call add(s:content, l:part)
        endfor

        " remove trailing new lines
        while len(s:content) > 0 && s:content[-1] == ""
            call remove(s:content, -1)
        endwhile

        " if response.timings
        if len(get(l:response, 'timings', {})) > 0
            let l:has_timing = v:true
            let l:timings = get(l:response, 'timings', {})

            let l:n_prompt    = get(l:timings, 'prompt_n', 0)
            let l:t_prompt_ms = get(l:timings, 'prompt_ms', 1)
            let l:s_prompt    = get(l:timings, 'prompt_per_second', 0)

            let l:n_gen    = get(l:timings, 'predicted_n', 0)
            let l:t_gen_ms = get(l:timings, 'predicted_ms', 1)
            let l:s_gen    = get(l:timings, 'predicted_per_second', 0)
        endif
    endif

    if len(s:content) == 0
        call add(s:content, "<| nothing to suggest |>")
        let l:can_accept = v:false
    endif

    let s:pos_dx = len(s:content[-1])
    let s:content[-1] .= l:line_cur_suffix

    " display virtual text with the suggestion
    let l:bufnr = bufnr('%')

    let s:id_vt_fim  = nvim_create_namespace('vt_fim')
    let s:id_vt_info = nvim_create_namespace('vt_info')

    call nvim_buf_set_extmark(l:bufnr, s:id_vt_fim, l:pos_y - 1, l:pos_x - 1, {
        \ 'virt_text': [[s:content[0], 'llama_hl_hint']],
        \ 'virt_text_win_col': l:pos_x == 1 ? 0 : virtcol('.')
        \ })

    call nvim_buf_set_extmark(l:bufnr, s:id_vt_fim, l:pos_y - 1, 0, {
        \ 'virt_lines': map(s:content[1:], {idx, val -> [[val, 'llama_hl_hint']]}),
        \ 'virt_text_win_col': virtcol('.')
        \ })

    " construct the info message:
    if l:has_timing
        let l:info = printf("prompt: %d (%.2f ms, %.2f t/s) | predict: %d (%.2f ms, %.2f t/s) | total: %f ms",
            \ l:n_prompt, l:t_prompt_ms, l:s_prompt,
            \ l:n_gen, l:t_gen_ms, l:s_gen,
            \ 1000.0 * reltimefloat(reltime(l:t_start))
            \ )

        call nvim_buf_set_extmark(l:bufnr, s:id_vt_info, l:pos_y - 1, l:pos_x - 1, {
            \ 'virt_text': [[l:info, 'llama_hl_info']],
            \ 'virt_text_pos': 'right_align',
            \ })
    endif

    " accept suggestion with Tab and reject it with any other key
    if l:can_accept
        inoremap <buffer> <Tab> <C-O>:call llama#accept_vt_fim()<CR>
    else
        inoremap <buffer> <Tab> <C-O>:call llama#cancel_vt_fim()<CR>
    endif

    for l:key in range(32, 127) + [8, 27]
        if l:key != 0x7C
            if l:key == 8
                execute 'inoremap <buffer> <Bs>    <C-O>:call llama#cancel_vt_fim()<CR><Bs>'
            elseif l:key == 27
                execute 'inoremap <buffer> <Esc>   <C-O>:call llama#cancel_vt_fim()<CR><Esc>'
            elseif l:key == 32
                execute 'inoremap <buffer> <Space> <C-O>:call llama#cancel_vt_fim()<CR><Space>'
            elseif l:key == 127
                execute 'inoremap <buffer> <Del>   <C-O>:call llama#cancel_vt_fim()<CR><Del>'
            else
                execute 'inoremap <buffer> ' . nr2char(l:key) . ' <C-O>:call llama#cancel_vt_fim()<CR>' . nr2char(l:key)
            endif
        endif
    endfor

    inoremap <buffer> <Up>    <C-O>:call llama#cancel_vt_fim()<CR><Up>
    inoremap <buffer> <Down>  <C-O>:call llama#cancel_vt_fim()<CR><Down>
    inoremap <buffer> <Left>  <C-O>:call llama#cancel_vt_fim()<CR><Left>
    inoremap <buffer> <Right> <C-O>:call llama#cancel_vt_fim()<CR><Right>
endfunction

function! llama#accept_vt_fim()
    let l:pos_x = col('.')
    let l:pos_y = line('.')

    let l:line_cur = getline('.')

    let l:pos0 = l:pos_x == len(l:line_cur) ? l:pos_x - 1 : l:pos_x - 2

    " insert the suggestion at the cursor location
    call setline(l:pos_y, l:line_cur[:l:pos0] . s:content[0])
    if len(s:content) > 1
        call append(l:pos_y, s:content[1:-1])
    endif

    " move the cursor to the end of the accepted text
    call cursor(l:pos_y + len(s:content) - 1, l:pos_x + s:pos_dx)

    call llama#cancel_vt_fim()
endfunction

function! llama#cancel_vt_fim()
    " clear the virtual text
    let l:bufnr = bufnr('%')

    call nvim_buf_clear_namespace(l:bufnr, s:id_vt_fim,  0, -1)
    call nvim_buf_clear_namespace(l:bufnr, s:id_vt_info, 0, -1)

    " remove the mappings
    iunmap <buffer> <Tab>

    for l:key in range(32, 127) + [8, 27]
        if l:key != 0x7C
            if l:key == 8
                execute 'iunmap <buffer> <Bs>'
            elseif l:key == 27
                execute 'iunmap <buffer> <Esc>'
            elseif l:key == 32
                execute 'iunmap <buffer> <Space>'
            elseif l:key == 127
                execute 'iunmap <buffer> <Del>'
            else
                execute 'iunmap <buffer> ' . nr2char(l:key)
            endif
        endif
    endfor

    iunmap <buffer> <Up>
    iunmap <buffer> <Down>
    iunmap <buffer> <Left>
    iunmap <buffer> <Right>
endfunction
