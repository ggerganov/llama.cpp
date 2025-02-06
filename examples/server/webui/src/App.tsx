import { HashRouter, Outlet, Route, Routes } from 'react-router';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import { AppContextProvider } from './utils/app.context';
import ChatScreen from './components/ChatScreen';

function App() {
  return (
    <HashRouter>
      <div className="flex flex-row drawer lg:drawer-open">
        <AppContextProvider>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/chat/:convId" element={<ChatScreen />} />
              <Route path="*" element={<ChatScreen />} />
            </Route>
          </Routes>
        </AppContextProvider>
      </div>
    </HashRouter>
  );
}

function AppLayout() {
  return (
    <>
      <Sidebar />
      <div className="chat-screen drawer-content grow flex flex-col h-screen w-screen mx-auto px-4">
        <Header />
        <Outlet />
      </div>
    </>
  );
}

export default App;
