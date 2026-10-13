/* 스캔앤고 사용자 로그인 (Supabase Auth, 이메일 + 비밀번호)
 *
 * 쓰는 법: 페이지에 <span id="auth-slot"></span> 을 두고
 *   supabase-js(UMD) → auth-config.js → auth.js 순서로 불러온다.
 * 로그인은 표시만 한다. 페이지 내용은 로그인과 상관없이 모두 보인다.
 */
(function () {
  var cfg = window.SCANGO_SUPABASE || {};
  var slot = document.getElementById('auth-slot');
  if (!slot) return;

  var ready = !!(cfg.url && cfg.anonKey && window.supabase);
  var sb = ready ? window.supabase.createClient(cfg.url, cfg.anonKey) : null;
  // 메일 링크(가입 확인·비밀번호 재설정)를 누르면 돌아올 주소: 지금 페이지 (쿼리·해시 제외)
  var here = location.origin + location.pathname;

  /* ----- 스타일 ----- */
  var css = document.createElement('style');
  css.textContent = [
    '.auth-btn{display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border-radius:8px;border:1px solid var(--line,#E2E8F0);background:var(--surface,#fff);color:var(--ink,#0F172A);font:600 .88rem var(--f-sans,sans-serif);cursor:pointer;white-space:nowrap;max-width:220px}',
    '.auth-btn:hover{border-color:var(--primary,#1D4ED8);color:var(--primary,#1D4ED8)}',
    '.auth-btn .em{overflow:hidden;text-overflow:ellipsis}',
    '.auth-dot{width:8px;height:8px;border-radius:50%;background:var(--good,#059669);flex:none}',
    'dialog.auth{margin:auto;inset:0;max-height:calc(100vh - 32px);overflow:auto;border:1px solid var(--line,#E2E8F0);border-radius:16px;padding:0;width:min(400px,calc(100vw - 32px));background:var(--surface,#fff);color:var(--ink,#0F172A);box-shadow:0 24px 60px rgba(15,23,42,.25)}',
    'dialog.auth::backdrop{background:rgba(15,23,42,.45)}',
    '.auth form{padding:26px 24px 22px;display:grid;gap:12px;font-family:var(--f-sans,sans-serif)}',
    '.auth h2{margin:0 0 4px;font-size:1.25rem;letter-spacing:-.02em;border:0;padding:0}',
    '.auth p{margin:0;color:var(--muted,#64748B);font-size:.9rem;line-height:1.6}',
    '.auth label{display:grid;gap:5px;font-size:.85rem;font-weight:600}',
    '.auth input{padding:11px 12px;border:1px solid var(--line,#E2E8F0);border-radius:8px;font:500 .95rem var(--f-sans,sans-serif);background:var(--bg,#F8FAFC);color:inherit}',
    '.auth input:focus{outline:2px solid var(--primary,#1D4ED8);outline-offset:1px}',
    '.auth .go{padding:12px;border:0;border-radius:8px;background:var(--primary,#1D4ED8);color:#fff;font:700 .95rem var(--f-sans,sans-serif);cursor:pointer}',
    '.auth .go:disabled{opacity:.6;cursor:wait}',
    '.auth .links{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:.85rem}',
    '.auth .links button,.auth .x{background:none;border:0;padding:0;color:var(--primary,#1D4ED8);font:600 .85rem var(--f-sans,sans-serif);cursor:pointer}',
    '.auth .x{position:absolute;top:14px;right:16px;color:var(--muted,#64748B);font-size:1.3rem;line-height:1}',
    '.auth .msg{padding:10px 12px;border-radius:8px;font-size:.88rem;line-height:1.5}',
    '.auth .msg.err{background:rgba(220,38,38,.1);color:#B91C1C}',
    '.auth .msg.ok{background:rgba(5,150,105,.1);color:#047857}',
    '.auth .me{padding:12px;border-radius:8px;background:var(--surface-2,#F1F5F9);font-weight:600;word-break:break-all}',
    '@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .auth .msg.err{color:#FCA5A5}:root:not([data-theme="light"]) .auth .msg.ok{color:#6EE7B7}}'
  ].join('\n');
  document.head.appendChild(css);

  /* ----- 상단 버튼 + 대화상자 ----- */
  var btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'auth-btn';
  slot.appendChild(btn);

  var dlg = document.createElement('dialog');
  dlg.className = 'auth';
  dlg.setAttribute('aria-label', '로그인');
  document.body.appendChild(dlg);
  dlg.addEventListener('click', function (e) { if (e.target === dlg) dlg.close(); });

  var user = null;
  var mode = 'login';  // login | signup | reset | newpw | account

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  function korean(err) {
    var m = (err && err.message) || String(err);
    if (/Invalid login credentials/i.test(m)) return '이메일 또는 비밀번호가 맞지 않습니다.';
    if (/Email not confirmed/i.test(m)) return '이메일 인증이 아직 안 됐습니다. 받은 편지함의 확인 메일 링크를 눌러 주세요.';
    if (/already registered|already been registered/i.test(m)) return '이미 가입된 이메일입니다. 로그인해 주세요.';
    if (/at least 6|Password should be/i.test(m)) return '비밀번호는 6자 이상이어야 합니다.';
    if (/rate limit|too many/i.test(m)) return '요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.';
    if (/invalid.*email|Unable to validate email/i.test(m)) return '이메일 주소 형식을 확인해 주세요.';
    if (/Failed to fetch|NetworkError/i.test(m)) return '서버에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요.';
    return m;
  }

  function renderButton() {
    if (user) {
      btn.innerHTML = '<span class="auth-dot" aria-hidden="true"></span><span class="em">' + esc(user.email) + '</span>';
      btn.title = '내 계정';
    } else {
      btn.textContent = '로그인';
      btn.title = '';
    }
  }

  function field(name, label, type, auto) {
    return '<label>' + label + '<input name="' + name + '" type="' + type + '" autocomplete="' + auto + '" required' +
      (type === 'password' ? ' minlength="6"' : '') + '></label>';
  }

  function render(msg, kind) {
    var h = '<form method="dialog" novalidate><button type="button" class="x" data-act="close" aria-label="닫기">×</button>';
    if (!ready) {
      h += '<h2>로그인 준비 중</h2><p>Supabase 연결 정보가 아직 설정되지 않았습니다. (<code>auth-config.js</code>)</p>';
    } else if (mode === 'account') {
      h += '<h2>내 계정</h2><div class="me">' + esc(user.email) + '</div>' +
        '<p>가입일: ' + new Date(user.created_at).toLocaleDateString('ko-KR') + '</p>' +
        '<button type="button" class="go" data-act="logout">로그아웃</button>' +
        '<div class="links"><button type="button" data-mode="newpw">비밀번호 바꾸기</button></div>';
    } else if (mode === 'signup') {
      h += '<h2>회원가입</h2><p>이메일로 가입하면 확인 메일이 갑니다.</p>' +
        field('email', '이메일', 'email', 'email') + field('password', '비밀번호 (6자 이상)', 'password', 'new-password') +
        field('password2', '비밀번호 확인', 'password', 'new-password') +
        '<button class="go" type="submit">가입하기</button>' +
        '<div class="links"><button type="button" data-mode="login">이미 계정이 있어요 → 로그인</button></div>';
    } else if (mode === 'reset') {
      h += '<h2>비밀번호 찾기</h2><p>가입한 이메일로 비밀번호 재설정 링크를 보냅니다.</p>' +
        field('email', '이메일', 'email', 'email') +
        '<button class="go" type="submit">재설정 메일 보내기</button>' +
        '<div class="links"><button type="button" data-mode="login">← 로그인으로</button></div>';
    } else if (mode === 'newpw') {
      h += '<h2>새 비밀번호 설정</h2>' +
        field('password', '새 비밀번호 (6자 이상)', 'password', 'new-password') +
        field('password2', '새 비밀번호 확인', 'password', 'new-password') +
        '<button class="go" type="submit">비밀번호 바꾸기</button>';
    } else {
      h += '<h2>로그인</h2><p>스캔앤고 계정으로 로그인합니다.</p>' +
        field('email', '이메일', 'email', 'email') + field('password', '비밀번호', 'password', 'current-password') +
        '<button class="go" type="submit">로그인</button>' +
        '<div class="links"><button type="button" data-mode="signup">회원가입</button><button type="button" data-mode="reset">비밀번호를 잊었어요</button></div>';
    }
    if (msg) h += '<div class="msg ' + (kind || 'err') + '" role="status">' + esc(msg) + '</div>';
    dlg.innerHTML = h + '</form>';
    var first = dlg.querySelector('input');
    if (first && dlg.open) first.focus();
  }

  // 입력한 값을 지우지 않고 안내 문구만 바꾼다
  function say(msg, kind) {
    var f = dlg.querySelector('form'), el = f.querySelector('.msg');
    if (!el) { el = document.createElement('div'); el.setAttribute('role', 'status'); f.appendChild(el); }
    el.className = 'msg ' + (kind || 'err');
    el.textContent = msg;
  }

  function open(m) {
    mode = m || (user ? 'account' : 'login');
    render();
    if (!dlg.open) dlg.showModal();
    var first = dlg.querySelector('input');
    if (first) first.focus();
  }

  btn.addEventListener('click', function () { open(); });

  dlg.addEventListener('click', function (e) {
    var t = e.target.closest('button');
    if (!t) return;
    if (t.dataset.mode) { mode = t.dataset.mode; render(); }
    else if (t.dataset.act === 'close') dlg.close();
    else if (t.dataset.act === 'logout') {
      sb.auth.signOut().then(function () { dlg.close(); });
    }
  });

  dlg.addEventListener('submit', async function (e) {
    e.preventDefault();
    var f = e.target, go = f.querySelector('.go');
    var email = f.email ? f.email.value.trim() : '';
    var pw = f.password ? f.password.value : '';
    if (f.email && !email) return say('이메일을 입력해 주세요.');
    if (f.password && pw.length < 6 && mode !== 'login') return say('비밀번호는 6자 이상이어야 합니다.');
    if (f.password2 && pw !== f.password2.value) return say('비밀번호 확인이 일치하지 않습니다.');
    go.disabled = true;
    try {
      var r;
      if (mode === 'login') {
        r = await sb.auth.signInWithPassword({ email: email, password: pw });
        if (r.error) throw r.error;
        dlg.close();
      } else if (mode === 'signup') {
        r = await sb.auth.signUp({ email: email, password: pw, options: { emailRedirectTo: here } });
        if (r.error) throw r.error;
        if (r.data.session) { dlg.close(); return; }  // 이메일 확인을 끈 프로젝트면 바로 로그인
        mode = 'login';
        render(email + ' 로 확인 메일을 보냈습니다. 메일의 링크를 누른 뒤 로그인해 주세요.', 'ok');
      } else if (mode === 'reset') {
        r = await sb.auth.resetPasswordForEmail(email, { redirectTo: here });
        if (r.error) throw r.error;
        mode = 'login';
        render('재설정 메일을 보냈습니다. 메일의 링크를 누르면 새 비밀번호를 정할 수 있습니다.', 'ok');
      } else if (mode === 'newpw') {
        r = await sb.auth.updateUser({ password: pw });
        if (r.error) throw r.error;
        mode = 'account';
        render('비밀번호를 바꿨습니다.', 'ok');
      }
    } catch (err) {
      say(korean(err));
    } finally {
      var g = dlg.querySelector('.go');
      if (g) g.disabled = false;
    }
  });

  renderButton();
  if (!ready) return;

  sb.auth.onAuthStateChange(function (event, session) {
    user = session ? session.user : null;
    renderButton();
    // 재설정 메일 링크로 들어오면 바로 새 비밀번호 화면
    if (event === 'PASSWORD_RECOVERY') setTimeout(function () { open('newpw'); }, 0);
  });
})();
