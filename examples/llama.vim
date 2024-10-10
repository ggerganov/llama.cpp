" sample config:
"
"   - Ctrl+F - trigger FIM completion manually
"
" run this once to initialise the plugin:
"
" :call llama#init()
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
    \ 'auto_fim':    v:true,
    \ 'stop':        ["\n"]
    \ }

let g:llama_config = get(g:, 'llama_config', s:default_config)

function! llama#init()
    let s:pos_x  = 0
    let s:pos_y  = 0
    let s:pos_x0 = 0 " pos_x corrected for end-of-line edge case

    let s:line_cur = ''

    let s:line_cur_prefix = ''
    let s:line_cur_suffix = ''

    let s:pos_dx = 0
    let s:content = []
    let s:can_accept = v:false

    let s:timer_fim = -1
    let s:t_fim_last  = reltime()
    let s:t_fim_start = reltime()

    let s:current_job = v:null

    augroup llama
        autocmd!
        autocmd InsertEnter * inoremap <buffer> <silent> <C-F> <C-O>:call llama#fim(v:false)<CR>
        autocmd InsertLeave * call llama#fim_cancel()
    augroup END

    silent! call llama#fim_cancel()
endfunction

function! llama#fim(is_auto) abort
    let s:t_fim_start = reltime()

    let s:content = []
    let s:can_accept = v:false

    let s:pos_x = col('.')
    let s:pos_y = line('.')
    let l:max_y = line('$')

    let l:lines_prefix = getline(max([1, s:pos_y - g:llama_config.n_prefix]), s:pos_y - 1)
    let l:lines_suffix = getline(s:pos_y + 1, min([l:max_y, s:pos_y + g:llama_config.n_suffix]))

    let s:line_cur = getline('.')

    let s:pos_x0 = s:pos_x == len(s:line_cur) ? s:pos_x : s:pos_x - 1

    let s:line_cur_prefix = strpart(s:line_cur, 0, s:pos_x0)
    let s:line_cur_suffix = strpart(s:line_cur, s:pos_x0)

    let l:prefix = ""
        \ . join(l:lines_prefix, "\n")
        \ . "\n"
        \ . s:line_cur_prefix

    let l:suffix = ""
        \ . s:line_cur_suffix
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

    let l:curl_command = printf(
        \ "curl --silent --no-buffer --request POST --url %s --header \"Content-Type: application/json\" --data %s",
        \ g:llama_config.endpoint, shellescape(l:request)
        \ )

    " send the request asynchronously
    let s:current_job = jobstart(l:curl_command, {
        \ 'on_stdout': function('s:fim_on_stdout'),
        \ 'on_exit': function('s:fim_on_exit'),
        \ 'stdout_buffered': v:true,
        \ 'is_auto': a:is_auto
        \ })
endfunction

function! llama#fim_accept()
    " insert the suggestion at the cursor location
    if s:can_accept && len(s:content) > 0
        call setline(s:pos_y, s:line_cur[:(s:pos_x0 - 1)] . s:content[0])
        if len(s:content) > 1
            call append(s:pos_y, s:content[1:-1])
        endif

        " move the cursor to the end of the accepted text
        call cursor(s:pos_y + len(s:content) - 1, s:pos_x + s:pos_dx)
    endif

    call llama#fim_cancel()
endfunction

function! llama#fim_cancel()
    if s:current_job != v:null
        call jobstop(s:current_job)
    endif

    " clear the virtual text
    let l:bufnr = bufnr('%')

    let l:id_vt_fim  = nvim_create_namespace('vt_fim')
    let l:id_vt_info = nvim_create_namespace('vt_info')

    call nvim_buf_clear_namespace(l:bufnr, l:id_vt_fim,  0, -1)
    call nvim_buf_clear_namespace(l:bufnr, l:id_vt_info, 0, -1)

    silent! iunmap <buffer> <Tab>
    silent! iunmap <buffer> <Esc>

    augroup llama_insert
        autocmd!
        if g:llama_config.auto_fim
            autocmd CursorMovedI * call s:fim_auto()
        endif
    augroup END
endfunction

function! s:fim_auto()
    if s:current_job != v:null
        call jobstop(s:current_job)
    endif

    if reltimefloat(reltime(s:t_fim_last)) < 0.001*250
        if s:timer_fim != -1
            call timer_stop(s:timer_fim)
            let s:timer_fim = -1
        endif
    endif

    let s:t_fim_last = reltime()
    let s:timer_fim = timer_start(250, {-> llama#fim(v:true)})
endfunction


function! s:fim_on_stdout(job_id, data, event) dict
    let l:raw = join(a:data, "\n")

    let s:can_accept = v:true
    let l:has_info   = v:false

    let l:n_prompt    = 0
    let l:t_prompt_ms = 1.0
    let l:s_prompt    = 0

    let l:n_gen    = 0
    let l:t_gen_ms = 1.0
    let l:s_gen    = 0

    if s:can_accept && v:shell_error
        if !self.is_auto
            call add(s:content, "<| curl error: is the server on? |>")
        endif
        let s:can_accept = v:false
    endif

    if s:can_accept && l:raw == ""
        if !self.is_auto
            call add(s:content, "<| empty response: is the server on? |>")
        endif
        let s:can_accept = v:false
    endif

    " get the generated suggestion
    if s:can_accept
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
            let l:has_info = v:true
            let l:timings  = get(l:response, 'timings', {})

            let l:n_prompt    = get(l:timings, 'prompt_n', 0)
            let l:t_prompt_ms = get(l:timings, 'prompt_ms', 1)
            let l:s_prompt    = get(l:timings, 'prompt_per_second', 0)

            let l:n_gen    = get(l:timings, 'predicted_n', 0)
            let l:t_gen_ms = get(l:timings, 'predicted_ms', 1)
            let l:s_gen    = get(l:timings, 'predicted_per_second', 0)
        endif
    endif

    if len(s:content) == 0
        if !self.is_auto
            call add(s:content, "<| nothing to suggest |>")
        endif
        let s:can_accept = v:false
    endif

    if len(s:content) == 0
        return
    endif

    let s:pos_dx = len(s:content[-1])
    let s:content[-1] .= s:line_cur_suffix

    call llama#fim_cancel()

    " display virtual text with the suggestion
    let l:bufnr = bufnr('%')

    let l:id_vt_fim  = nvim_create_namespace('vt_fim')
    let l:id_vt_info = nvim_create_namespace('vt_info')

    " construct the info message:
    if l:has_info
        " prefix the info string with whitespace in order to offset it to the right of the fim overlay
        let l:prefix = repeat(' ', len(s:content[0]) - len(s:line_cur_suffix) + 3)

        let l:info = printf("%s | prompt: %d (%.2f ms, %.2f t/s) | predict: %d (%.2f ms, %.2f t/s) | total: %f.2 ms",
            \ l:prefix,
            \ l:n_prompt, l:t_prompt_ms, l:s_prompt,
            \ l:n_gen, l:t_gen_ms, l:s_gen,
            \ 1000.0 * reltimefloat(reltime(s:t_fim_start))
            \ )

        call nvim_buf_set_extmark(l:bufnr, l:id_vt_info, s:pos_y - 1, s:pos_x - 1, {
            \ 'virt_text': [[l:info, 'llama_hl_info']],
            \ 'virt_text_pos': 'eol',
            \ })
    endif

    call nvim_buf_set_extmark(l:bufnr, l:id_vt_fim, s:pos_y - 1, s:pos_x - 1, {
        \ 'virt_text': [[s:content[0], 'llama_hl_hint']],
        \ 'virt_text_win_col': s:pos_x == len(s:line_cur) ? virtcol('.') : virtcol('.') - 1
        \ })

    call nvim_buf_set_extmark(l:bufnr, l:id_vt_fim, s:pos_y - 1, 0, {
        \ 'virt_lines': map(s:content[1:], {idx, val -> [[val, 'llama_hl_hint']]}),
        \ 'virt_text_win_col': virtcol('.')
        \ })

    " setup accept/cancel events
    inoremap <buffer> <Tab> <C-O>:call llama#fim_accept()<CR>
    inoremap <buffer> <Esc> <C-O>:call llama#fim_cancel()<CR><Esc>

    augroup llama_insert
        autocmd!
        autocmd CursorMovedI * call llama#fim_cancel()
    augroup END
endfunction

function! s:fim_on_exit(job_id, exit_code, event) dict
    if a:exit_code != 0
        echom "Job failed with exit code: " . a:exit_code
    endif

    let s:current_job = v:null
endfunction
