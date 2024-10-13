" LLM-based text completion using llama.cpp
"
" requires:
"
"   - neovim
"   - curl
"   - llama.cpp server instance
"   - FIM-compatible model
"
" sample config:
"
"   - Tab       - accept the current suggestion
"   - Shift+Tab - accept just the first line of the segguestion
"   - Ctrl+F    - trigger FIM completion manually
"
" make symlink or copy this file to ~/.config/nvim/autoload/llama.vim
"
" start the llama.cpp server with a FIM-compatible model. for example:
"
"   $ llama-server -m {model.gguf} --port 8012 -ngl 99 -fa --ubatch-size 1024 --batch-size 2048 --cache-reuse 512
"
"   --batch-size [512, model max context]
"
"     adjust the batch size to control how much of the provided context will be used during the inference
"     lower values will use smaller part of the context around the cursor, which will result in faster processing
"
"   --ubatch-size [64, 2048]
"
"     chunks the batch into smaller chunks for faster processing
"     depends on the specific hardware. use llama-bench to profile and determine the best size
"
" run this once to initialise llama.vim:
"
"   :call llama#init()
"

" color of the suggested text
highlight llama_hl_hint guifg=#ff772f
highlight llama_hl_info guifg=#77ff2f

" general parameters:
"
"   endpoint:         llama.cpp server endpoint
"   n_prefix:         number of lines before the cursor location to include in the prefix
"   n_suffix:         number of lines after  the cursor location to include in the suffix
"   n_predict:        max number of tokens to predict
"   t_max_prompt_ms:  max alloted time for the prompt generation (TODO: not yet supported)
"   t_max_predict_ms: max alloted time for the prediction
"   show_info:        show extra info about the inference (0 - disabled, 1 - statusline, 2 - inline)
"   auto_fim:         trigger FIM completion automatically on cursor movement
"
" ring buffer of chunks, accumulated with time upon:
"
"  - completion request
"  - yank
"  - reading a file
"
" ring context parameters:
"
"   ring_n_chunks:    max number of chunks to pass as extra context to the server (0 to disable)
"   ring_chunk_size:  max size of the chunks (in number of lines)
"   ring_scope:       the range around the cursor position (in number of lines) for gathering chunks
"
let s:default_config = {
    \ 'endpoint':         'http://127.0.0.1:8012/infill',
    \ 'n_prefix':         128,
    \ 'n_suffix':         128,
    \ 'n_predict':        64,
    \ 't_max_prompt_ms':  500,
    \ 't_max_predict_ms': 200,
    \ 'show_info':        2,
    \ 'auto_fim':         v:true,
    \ 'ring_n_chunks':    32,
    \ 'ring_chunk_size':  64,
    \ 'ring_scope':       1024,
    \ }

let g:llama_config = get(g:, 'llama_config', s:default_config)

function! s:rand(i0, i1) abort
    return a:i0 + rand() % (a:i1 - a:i0 + 1)
endfunction

function! llama#init()
    if !executable('curl')
        echohl WarningMsg
        echo 'llama.vim requires the "curl" command to be available'
        echohl None
        return
    endif

    let s:pos_x  = 0 " cursor position upon start of completion
    let s:pos_y  = 0
    let s:pos_x0 = 0 " pos_x corrected for end-of-line edge case

    let s:line_cur = ''

    let s:line_cur_prefix = ''
    let s:line_cur_suffix = ''

    let s:ring_chunks = []
    let s:ring_n_evict = 0

    let s:pos_y_pick = -9999 " last y where we picked a chunk
    let s:pos_dx = 0
    let s:content = []
    let s:can_accept = v:false

    let s:timer_fim = -1
    let s:t_fim_last  = reltime()
    let s:t_fim_start = reltime()

    let s:current_job = v:null

    augroup llama
        autocmd!
        autocmd InsertEnter    * inoremap <buffer> <silent> <C-F> <C-O>:call llama#fim(v:false)<CR>
        autocmd InsertLeavePre * call llama#fim_cancel()

        autocmd CursorMoved    * call llama#fim_cancel()

        autocmd TextYankPost   * if v:event.operator ==# 'y' | call s:pick_chunk(v:event.regcontents, v:false) | endif

        " gather chunks upon entering/leaving a buffer
        autocmd BufEnter       * call timer_start(100, {-> s:pick_chunk(getline(max([1, line('.') - g:llama_config.ring_chunk_size/2]), min([line('.') + g:llama_config.ring_chunk_size/2, line('$')])), v:true)})
        autocmd BufLeave       * call                      s:pick_chunk(getline(max([1, line('.') - g:llama_config.ring_chunk_size/2]), min([line('.') + g:llama_config.ring_chunk_size/2, line('$')])), v:true)
    augroup END

    silent! call llama#fim_cancel()
endfunction

" TODO: figure out something better
function! s:chunk_sim(c0, c1)
    let l:lines0 = len(a:c0)
    let l:lines1 = len(a:c1)

    let l:common = 0

    for l:line0 in a:c0
        for l:line1 in a:c1
            if l:line0 == l:line1
                let l:common += 1
                break
            endif
        endfor
    endfor

    return 2.0 * l:common / (l:lines0 + l:lines1)
endfunction

function! s:pick_chunk(text, no_mod)
    " do not pick chunks from buffers with pending changes or buffers that are not files
    if a:no_mod && (getbufvar(bufnr('%'), '&modified') || !buflisted(bufnr('%')) || !filereadable(expand('%')))
        return
    endif

    if g:llama_config.ring_n_chunks <= 0
        return
    endif

    if len(a:text) < 3
        return
    endif

    if len(a:text) + 1 < g:llama_config.ring_chunk_size
        let l:chunk = a:text
    else
        let l:l0 = s:rand(0, max([0, len(a:text) - g:llama_config.ring_chunk_size]))
        let l:l1 = min([l:l0 + g:llama_config.ring_chunk_size, len(a:text)])

        let l:chunk = a:text[l:l0:l:l1]
    endif

    let l:chunk_str = join(l:chunk, "\n")

    " check if this chunk is already added
    let l:exist = v:false
    for i in range(len(s:ring_chunks))
        if s:ring_chunks[i].data == l:chunk
            let l:exist = v:true
            break
        endif
    endfor

    if l:exist
        return
    endif

    " evict chunks that are very similar to the new one
    for i in range(len(s:ring_chunks) - 1, 0, -1)
        if s:chunk_sim(s:ring_chunks[i].data, l:chunk) > 0.9
            call remove(s:ring_chunks, i)
            let s:ring_n_evict += 1
        endif
    endfor

    if len(s:ring_chunks) == g:llama_config.ring_n_chunks
        call remove(s:ring_chunks, 0)
    endif

    call add(s:ring_chunks, {'data': l:chunk, 'str': l:chunk_str, 'time': reltime()})
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

    " TODO: per-file location
    let l:delta_y = abs(s:pos_y - s:pos_y_pick)

    " only gather chunks if the cursor has moved a lot
    if a:is_auto && l:delta_y > 32
        " pick a prefix chunk
        call s:pick_chunk(getline(max([1, s:pos_y - g:llama_config.ring_scope]), max([1, s:pos_y - g:llama_config.n_prefix])), v:false)

        " pick a suffix chunk
        call s:pick_chunk(getline(min([l:max_y, s:pos_y + g:llama_config.n_suffix]), min([l:max_y, s:pos_y + g:llama_config.ring_scope])), v:false)

        let s:pos_y_pick = s:pos_y
    endif

    " array of strings
    let l:extra_context = []
    for l:chunk in s:ring_chunks
        call add(l:extra_context, l:chunk.str)
    endfor

    let l:request = json_encode({
        \ 'prompt':           "",
        \ 'input_prefix':     l:prefix,
        \ 'input_suffix':     l:suffix,
        \ 'n_predict':        g:llama_config.n_predict,
        \ 'penalty_last_n':   0,
        \ 'top_k':            100,
        \ 'stream':           v:false,
        \ 'samplers':         ["top_k", "infill"],
        \ 'cache_prompt':     v:true,
        \ 'extra_context':    l:extra_context,
        \ 't_max_prompt_ms':  g:llama_config.t_max_prompt_ms,
        \ 't_max_predict_ms': g:llama_config.t_max_predict_ms
        \ })

    let l:curl_command = printf(
        \ "curl --silent --no-buffer --request POST --url %s --header \"Content-Type: application/json\" --data %s",
        \ g:llama_config.endpoint, shellescape(l:request)
        \ )

    " send the request asynchronously
    let s:current_job = jobstart(l:curl_command, {
        \ 'on_stdout': function('s:fim_on_stdout'),
        \ 'on_exit':   function('s:fim_on_exit'),
        \ 'stdout_buffered': v:true,
        \ 'is_auto': a:is_auto
        \ })

    " this trick is needed to avoid the cursor shifting upon C-O when at the end of the line
    if !a:is_auto
        augroup llama_insert
            autocmd!
        augroup END

        if g:llama_config.auto_fim
            call timer_start(0, {-> s:fim_auto_enable()})
        endif
    endif
endfunction

" if first_line == v:true accept only the first line of the response
function! llama#fim_accept(first_line)
    " insert the suggestion at the cursor location
    if s:can_accept && len(s:content) > 0
        call setline(s:pos_y, s:line_cur[:(s:pos_x0 - 1)] . s:content[0])
        if len(s:content) > 1
            if !a:first_line
                call append(s:pos_y, s:content[1:-1])
            endif
        endif

        " move the cursor to the end of the accepted text
        if !a:first_line
            call cursor(s:pos_y + len(s:content) - 1, s:pos_x + s:pos_dx)
        else
            call cursor(s:pos_y, s:pos_x + len(s:content[0]) - 1)
        endif
    endif

    call llama#fim_cancel()
endfunction

function! llama#fim_cancel()
    if s:current_job != v:null
        call jobstop(s:current_job)
    endif

    if s:timer_fim != -1
        call timer_stop(s:timer_fim)
        let s:timer_fim = -1
    endif

    " clear the virtual text
    let l:bufnr = bufnr('%')

    let l:id_vt_fim  = nvim_create_namespace('vt_fim')
    let l:id_vt_info = nvim_create_namespace('vt_info')

    call nvim_buf_clear_namespace(l:bufnr, l:id_vt_fim,  0, -1)
    call nvim_buf_clear_namespace(l:bufnr, l:id_vt_info, 0, -1)

    " remove the mappings
    silent! iunmap <buffer> <Tab>
    silent! iunmap <buffer> <S-Tab>
    silent! iunmap <buffer> <Esc>

    augroup llama_insert
        autocmd!
    augroup END

    if g:llama_config.auto_fim
        call s:fim_auto_enable()
    endif
endfunction

function! s:fim_auto_enable()
    augroup llama_insert
        autocmd CursorMovedI * call s:fim_auto()
    augroup END
endfunction

" auto-start a fim job a short time after the cursor has moved
" if there is already a job queued - cancel it
function! s:fim_auto()
    if s:current_job != v:null
        call jobstop(s:current_job)
    endif

    " TODO: when job cancellation is implemented on the server, reduce these timeouts
    if reltimefloat(reltime(s:t_fim_last)) < 500*0.001
        if s:timer_fim != -1
            call timer_stop(s:timer_fim)
            let s:timer_fim = -1
        endif
    endif

    let s:t_fim_last = reltime()
    let s:timer_fim = timer_start(500, {-> llama#fim(v:true)})
endfunction

" callback that processes the result from the server
function! s:fim_on_stdout(job_id, data, event) dict
    let l:raw = join(a:data, "\n")
    if len(l:raw) == 0
        return
    endif

    let s:can_accept = v:true
    let l:has_info   = v:false

    if s:can_accept && v:shell_error
        if !self.is_auto
            call add(s:content, "<| curl error: is the server on? |>")
        endif
        let s:can_accept = v:false
    endif

    let l:n_prompt    = 0
    let l:t_prompt_ms = 1.0
    let l:s_prompt    = 0

    let l:n_predict    = 0
    let l:t_predict_ms = 1.0
    let l:s_predict    = 0

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

        let l:generation_settings = get(l:response, 'generation_settings', {})
        let l:n_ctx = get(l:generation_settings, 'n_ctx', 0)

        let l:n_cached = get(l:response, 'tokens_cached', 0)

        " if response.timings is available
        if len(get(l:response, 'timings', {})) > 0
            let l:has_info = v:true
            let l:timings  = get(l:response, 'timings', {})

            let l:n_prompt    = get(l:timings, 'prompt_n', 0)
            let l:t_prompt_ms = get(l:timings, 'prompt_ms', 1)
            let l:s_prompt    = get(l:timings, 'prompt_per_second', 0)

            let l:n_predict    = get(l:timings, 'predicted_n', 0)
            let l:t_predict_ms = get(l:timings, 'predicted_ms', 1)
            let l:s_predict    = get(l:timings, 'predicted_per_second', 0)
        endif
    endif

    if len(s:content) == 0
        if !self.is_auto
            call add(s:content, "<| EOT |>")
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

    " construct the info message and display it to the right of the current line
    if g:llama_config.show_info > 0 && l:has_info
        " prefix the info string with whitespace in order to offset it to the right of the fim overlay
        let l:prefix = repeat(' ', len(s:content[0]) - len(s:line_cur_suffix) + 3)

        let l:info = printf("%s | context: %d / %d / %d / %d | prompt: %d (%.2f ms, %.2f t/s) | predict: %d (%.2f ms, %.2f t/s) | total: %.2f ms",
            \ g:llama_config.show_info == 2 ? l:prefix : '',
            \ l:n_cached,  l:n_ctx, len(s:ring_chunks), s:ring_n_evict,
            \ l:n_prompt,  l:t_prompt_ms,  l:s_prompt,
            \ l:n_predict, l:t_predict_ms, l:s_predict,
            \ 1000.0 * reltimefloat(reltime(s:t_fim_start))
            \ )

        if g:llama_config.show_info == 1
            let &statusline = l:info
        elseif g:llama_config.show_info == 2
            call nvim_buf_set_extmark(l:bufnr, l:id_vt_info, s:pos_y - 1, s:pos_x - 1, {
                \ 'virt_text': [[l:info, 'llama_hl_info']],
                \ 'virt_text_pos': 'eol',
                \ })
        endif
    endif

    " display the suggestion
    call nvim_buf_set_extmark(l:bufnr, l:id_vt_fim, s:pos_y - 1, s:pos_x - 1, {
        \ 'virt_text': [[s:content[0], 'llama_hl_hint']],
        \ 'virt_text_win_col': virtcol('.') - 1
        \ })

    call nvim_buf_set_extmark(l:bufnr, l:id_vt_fim, s:pos_y - 1, 0, {
        \ 'virt_lines': map(s:content[1:], {idx, val -> [[val, 'llama_hl_hint']]}),
        \ 'virt_text_win_col': virtcol('.')
        \ })

    " setup accept/cancel events
    inoremap <buffer> <Tab>   <C-O>:call llama#fim_accept(v:false)<CR>
    inoremap <buffer> <S-Tab> <C-O>:call llama#fim_accept(v:true)<CR>

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
