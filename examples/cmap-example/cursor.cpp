// just trying to get the cursor position

#include <cstdlib>

struct CursorPos {
  int x;
  int y;
};

static CursorPos getCursorPos() {

  // Get text cursor position
  auto cursorPos = getCursorPos();

  // Assign to struct
  CursorPos pos;
  pos.x = cursorPos.x;
  pos.y = cursorPos.y;

  return pos;
}

int main() {
    CursorPos cursor = getCursorPos();
    printf("The x co-ordinate of the cursor is %zu\n; the y co-ordinate of the cursor is %zu\n", cursor.x, cursor.y);
}
