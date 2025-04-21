{{/* _spiffe.tpl – helper that renders a SPIRE agent side‑car when enabled */}}
{{- define "af.spiffeSidecar" -}}
{{- if .Values.spiffe.enabled }}
- name: spire-agent
  image: ghcr.io/spiffe/spire-agent:1.8
  imagePullPolicy: IfNotPresent
  securityContext:
    runAsUser: 1337
    runAsGroup: 1337
    privileged: false
    allowPrivilegeEscalation: false
  volumeMounts:
    - name: spire-socket
      mountPath: /run/spire
  env:
    - name: SPIFFE_ENDPOINT_SOCKET
      value: /run/spire/sock
{{- end }}
{{- end }}
