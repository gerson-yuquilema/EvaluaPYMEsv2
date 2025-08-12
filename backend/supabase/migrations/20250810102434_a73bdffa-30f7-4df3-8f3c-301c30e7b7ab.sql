-- Enable required extension for UUIDs
create extension if not exists "pgcrypto";

-- 1) Roles and RBAC helpers
create type if not exists public.app_role as enum ('admin', 'analyst', 'applicant');

create table if not exists public.user_roles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  role public.app_role not null,
  unique(user_id, role)
);

alter table public.user_roles enable row level security;

create or replace function public.has_role(_user_id uuid, _role public.app_role)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.user_roles
    where user_id = _user_id and role = _role
  );
$$;

-- Policies for user_roles
create policy if not exists "user_roles_select_own_or_admin"
  on public.user_roles for select to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(),'admin'));

create policy if not exists "user_roles_insert_self_applicant_or_admin"
  on public.user_roles for insert to authenticated
  with check ((user_id = auth.uid() and role = 'applicant') or public.has_role(auth.uid(),'admin'));

create policy if not exists "user_roles_update_admin_only"
  on public.user_roles for update to authenticated
  using (public.has_role(auth.uid(),'admin'))
  with check (public.has_role(auth.uid(),'admin'));

create policy if not exists "user_roles_delete_admin_only"
  on public.user_roles for delete to authenticated
  using (public.has_role(auth.uid(),'admin'));

-- 2) Profiles table and trigger to auto-create on signup
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  avatar_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

-- Common updated_at trigger
create or replace function public.update_updated_at_column()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, display_name, avatar_url)
  values (new.id, new.raw_user_meta_data ->> 'display_name', new.raw_user_meta_data ->> 'avatar_url')
  on conflict (id) do nothing;
  return new;
end;
$$;

do $$
begin
  if not exists (
    select 1 from pg_trigger where tgname = 'on_auth_user_created'
  ) then
    create trigger on_auth_user_created
      after insert on auth.users
      for each row execute procedure public.handle_new_user();
  end if;
end $$;

create policy if not exists "profiles_select_self_or_admin_analyst"
  on public.profiles for select to authenticated
  using (id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'));

create policy if not exists "profiles_insert_self"
  on public.profiles for insert to authenticated
  with check (id = auth.uid());

create policy if not exists "profiles_update_self_or_admin_analyst"
  on public.profiles for update to authenticated
  using (id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
  with check (id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'));

-- 3) Core business tables
create table if not exists public.empresas (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  ruc text not null,
  razon_social text not null,
  actividad_economica text,
  sector text,
  estado_contribuyente text,
  anios_operacion smallint,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (ruc)
);

alter table public.empresas enable row level security;

create index if not exists idx_empresas_user on public.empresas(user_id);

create trigger if not exists update_empresas_updated_at
before update on public.empresas
for each row execute function public.update_updated_at_column();

create policy if not exists "empresas_select_owner_or_admin_analyst"
  on public.empresas for select to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'));

create policy if not exists "empresas_insert_self_or_admin_analyst"
  on public.empresas for insert to authenticated
  with check (user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'));

create policy if not exists "empresas_update_owner_or_admin_analyst"
  on public.empresas for update to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
  with check (user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'));

create policy if not exists "empresas_delete_owner_or_admin"
  on public.empresas for delete to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(),'admin'));

-- 4) Documents metadata table
create table if not exists public.documentos (
  id uuid primary key default gen_random_uuid(),
  empresa_id uuid not null references public.empresas(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  tipo text, -- factura, estado_financiero, referencia, etc.
  bucket text not null default 'documentos',
  path text not null, -- storage path
  original_name text,
  content_type text,
  size_bytes bigint,
  status text default 'uploaded', -- uploaded | parsed | failed
  parsed_json jsonb,
  created_at timestamptz not null default now()
);

alter table public.documentos enable row level security;

create index if not exists idx_documentos_empresa on public.documentos(empresa_id);
create index if not exists idx_documentos_user on public.documentos(user_id);

create policy if not exists "documentos_select_owner_or_admin_analyst"
  on public.documentos for select to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'));

create policy if not exists "documentos_insert_self"
  on public.documentos for insert to authenticated
  with check (user_id = auth.uid());

create policy if not exists "documentos_update_owner_or_admin_analyst"
  on public.documentos for update to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
  with check (user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'));

create policy if not exists "documentos_delete_owner_or_admin"
  on public.documentos for delete to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(),'admin'));

-- 5) Financial metrics
create table if not exists public.metricas_financieras (
  id uuid primary key default gen_random_uuid(),
  empresa_id uuid not null references public.empresas(id) on delete cascade,
  periodo_inicio date,
  periodo_fin date,
  ventas numeric,
  flujo_caja numeric,
  activos numeric,
  deudas numeric,
  liquidez numeric,
  rentabilidad numeric,
  created_at timestamptz not null default now()
);

alter table public.metricas_financieras enable row level security;
create index if not exists idx_metricas_empresa on public.metricas_financieras(empresa_id);

create policy if not exists "metricas_select_owner_or_admin_analyst"
  on public.metricas_financieras for select to authenticated
  using (
    exists (
      select 1 from public.empresas e
      where e.id = metricas_financieras.empresa_id
        and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "metricas_insert_owner_or_admin_analyst"
  on public.metricas_financieras for insert to authenticated
  with check (
    exists (
      select 1 from public.empresas e
      where e.id = empresa_id and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "metricas_update_owner_or_admin_analyst"
  on public.metricas_financieras for update to authenticated
  using (
    exists (
      select 1 from public.empresas e
      where e.id = metricas_financieras.empresa_id
        and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  )
  with check (
    exists (
      select 1 from public.empresas e
      where e.id = empresa_id and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "metricas_delete_owner_or_admin"
  on public.metricas_financieras for delete to authenticated
  using (
    exists (
      select 1 from public.empresas e
      where e.id = metricas_financieras.empresa_id
        and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin'))
    )
  );

-- 6) Social profiles
create table if not exists public.redes_sociales (
  id uuid primary key default gen_random_uuid(),
  empresa_id uuid not null references public.empresas(id) on delete cascade,
  red text, -- facebook, instagram, x, google, etc.
  url text not null,
  handle text,
  estado text default 'pendiente',
  created_at timestamptz not null default now()
);

alter table public.redes_sociales enable row level security;
create index if not exists idx_redes_empresa on public.redes_sociales(empresa_id);

create policy if not exists "redes_select_owner_or_admin_analyst"
  on public.redes_sociales for select to authenticated
  using (
    exists (
      select 1 from public.empresas e where e.id = redes_sociales.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "redes_insert_owner_or_admin_analyst"
  on public.redes_sociales for insert to authenticated
  with check (
    exists (
      select 1 from public.empresas e where e.id = empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "redes_update_owner_or_admin_analyst"
  on public.redes_sociales for update to authenticated
  using (
    exists (
      select 1 from public.empresas e where e.id = redes_sociales.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  )
  with check (
    exists (
      select 1 from public.empresas e where e.id = empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "redes_delete_owner_or_admin"
  on public.redes_sociales for delete to authenticated
  using (
    exists (
      select 1 from public.empresas e where e.id = redes_sociales.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin'))
    )
  );

-- 7) Social insights
create table if not exists public.insights_sociales (
  id uuid primary key default gen_random_uuid(),
  empresa_id uuid not null references public.empresas(id) on delete cascade,
  collected_at timestamptz not null default now(),
  reviews_count integer,
  comments_count integer,
  interactions_count integer,
  sentiment_score numeric, -- -1 a 1
  reputacion_digital numeric, -- 0 a 100
  fuente text
);

alter table public.insights_sociales enable row level security;
create index if not exists idx_insights_empresa on public.insights_sociales(empresa_id);

create policy if not exists "insights_select_owner_or_admin_analyst"
  on public.insights_sociales for select to authenticated
  using (
    exists (
      select 1 from public.empresas e where e.id = insights_sociales.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "insights_insert_owner_or_admin_analyst"
  on public.insights_sociales for insert to authenticated
  with check (
    exists (
      select 1 from public.empresas e where e.id = empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "insights_update_owner_or_admin_analyst"
  on public.insights_sociales for update to authenticated
  using (
    exists (
      select 1 from public.empresas e where e.id = insights_sociales.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  )
  with check (
    exists (
      select 1 from public.empresas e where e.id = empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "insights_delete_owner_or_admin"
  on public.insights_sociales for delete to authenticated
  using (
    exists (
      select 1 from public.empresas e where e.id = insights_sociales.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin'))
    )
  );

-- 8) Credit scores
create table if not exists public.scores_credito (
  id uuid primary key default gen_random_uuid(),
  empresa_id uuid not null references public.empresas(id) on delete cascade,
  score numeric not null check (score >= 0 and score <= 100),
  banda text, -- verde/amarillo/rojo
  monto_recomendado numeric,
  explicacion text,
  created_at timestamptz not null default now()
);

alter table public.scores_credito enable row level security;
create index if not exists idx_scores_empresa on public.scores_credito(empresa_id);

create policy if not exists "scores_select_owner_or_admin_analyst"
  on public.scores_credito for select to authenticated
  using (
    exists (
      select 1 from public.empresas e where e.id = scores_credito.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "scores_insert_owner_or_admin_analyst"
  on public.scores_credito for insert to authenticated
  with check (
    exists (
      select 1 from public.empresas e where e.id = empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "scores_update_owner_or_admin_analyst"
  on public.scores_credito for update to authenticated
  using (
    exists (
      select 1 from public.empresas e where e.id = scores_credito.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  )
  with check (
    exists (
      select 1 from public.empresas e where e.id = empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "scores_delete_owner_or_admin"
  on public.scores_credito for delete to authenticated
  using (
    exists (
      select 1 from public.empresas e where e.id = scores_credito.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin'))
    )
  );

-- 9) Extend existing reviews table to link with empresas and secure it
alter table public.reviews add column if not exists empresa_id uuid references public.empresas(id) on delete cascade;

alter table public.reviews enable row level security;
create index if not exists idx_reviews_empresa on public.reviews(empresa_id);

create policy if not exists "reviews_select_owner_or_admin_analyst"
  on public.reviews for select to authenticated
  using (
    empresa_id is null or exists (
      select 1 from public.empresas e where e.id = reviews.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "reviews_insert_owner_or_admin_analyst"
  on public.reviews for insert to authenticated
  with check (
    empresa_id is null or exists (
      select 1 from public.empresas e where e.id = empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "reviews_update_owner_or_admin_analyst"
  on public.reviews for update to authenticated
  using (
    empresa_id is null or exists (
      select 1 from public.empresas e where e.id = reviews.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  )
  with check (
    empresa_id is null or exists (
      select 1 from public.empresas e where e.id = empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin') or public.has_role(auth.uid(),'analyst'))
    )
  );

create policy if not exists "reviews_delete_owner_or_admin"
  on public.reviews for delete to authenticated
  using (
    empresa_id is null or exists (
      select 1 from public.empresas e where e.id = reviews.empresa_id
      and (e.user_id = auth.uid() or public.has_role(auth.uid(),'admin'))
    )
  );

-- 10) Storage bucket and policies for documents
insert into storage.buckets (id, name, public)
values ('documentos', 'documentos', false)
on conflict (id) do nothing;

-- Policies on storage.objects for documentos bucket
create policy if not exists "storage_documentos_select_own_or_admin_analyst"
  on storage.objects for select to authenticated
  using (
    bucket_id = 'documentos' and (
      auth.uid()::text = (storage.foldername(name))[1]
      or public.has_role(auth.uid(),'admin')
      or public.has_role(auth.uid(),'analyst')
    )
  );

create policy if not exists "storage_documentos_insert_self"
  on storage.objects for insert to authenticated
  with check (
    bucket_id = 'documentos' and auth.uid()::text = (storage.foldername(name))[1]
  );

create policy if not exists "storage_documentos_update_own_or_admin_analyst"
  on storage.objects for update to authenticated
  using (
    bucket_id = 'documentos' and (
      auth.uid()::text = (storage.foldername(name))[1]
      or public.has_role(auth.uid(),'admin')
      or public.has_role(auth.uid(),'analyst')
    )
  )
  with check (
    bucket_id = 'documentos' and (
      auth.uid()::text = (storage.foldername(name))[1]
      or public.has_role(auth.uid(),'admin')
      or public.has_role(auth.uid(),'analyst')
    )
  );

create policy if not exists "storage_documentos_delete_own_or_admin"
  on storage.objects for delete to authenticated
  using (
    bucket_id = 'documentos' and (
      auth.uid()::text = (storage.foldername(name))[1]
      or public.has_role(auth.uid(),'admin')
    )
  );