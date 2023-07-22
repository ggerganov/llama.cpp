function! Llm()

  let url = "http://127.0.0.1:8080/completion"

  " Save the current cursor position
  let save_cursor = getpos('.')

  silent! %s/\n/\\n/g
  silent! %s/\t/\\t/g
  silent! %s/\\n$//

  " Get the content of the current buffer
  let buffer_content = join(getline(1, '$'), "\n")

  " Replace true newlines with "\n"
  let buffer_content = substitute(buffer_content, '\n', '\\n', 'g')

  " Trim leading/trailing whitespace
  let buffer_content = substitute(buffer_content, '^\s\+', '', '')
  let buffer_content = substitute(buffer_content, '\s\+$', '', '')

  " Create the JSON payload
  " can't escape backslash, \n gets replaced as \\n
  let json_payload = '{"prompt":"' . escape(buffer_content, '"/') . '","temp":0.72,"top_k":100,"top_p":0.73,"repeat_penalty":1.100000023841858,"n_predict":10,"stream":false}'

  let prompt_tmpfile = tempname()
  let response_tmpfile = tempname()
  call writefile([json_payload], prompt_tmpfile)

  " Define the curl command
  let curl_command = 'curl -k -s -X POST -H "Content-Type: application/json" -o ' . shellescape(response_tmpfile) . ' -d @' . shellescape(prompt_tmpfile) . ' ' . url
  silent execute '!'.curl_command

  let response = join(readfile(response_tmpfile), '')
  let start_marker = '{"content":"'
  let end_marker = '","generation_settings'
  let content_start = stridx(response, start_marker) + len(start_marker)
  let content_end = stridx(response, end_marker, content_start)

  " Extract the content field from the response
  let content = strpart(response, content_start, content_end - content_start)

  " Insert the content at the cursor position
  call setline(line('.'), getline('.') . content)

  " Replace newline "\n" strings with actual newlines in the content
  silent! %s/\\n/\r/g
  " and tabs
  silent! %s/\\t/\t/g
  " and quote marks for C sources
  silent! %s/\\"/\"/g

  " Remove the temporary file
  call delete(prompt_tmpfile)
  call delete(response_tmpfile)
endfunction

command! Llm call Llm()
