"use client"

import { useState, useEffect, useRef } from "react"
import { RiSendPlaneFill, RiSparklingLine, RiLoader2Fill } from "@remixicon/react"
import ReactMarkdown from "react-markdown"
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerDescription,
  DrawerBody,
  DrawerFooter,
} from "@/components/Drawer"
import { Button } from "@/components/Button"
import { Input } from "@/components/Input"
import { cx } from "@/lib/utils"

interface Message {
  role: "user" | "assistant"
  content: string
}

interface ToolCallStatus {
  name: string
  args: Record<string, unknown>
}

// Human-readable tool names
const TOOL_DISPLAY_NAMES: Record<string, string> = {
  get_stock_news: "Fetching stock news",
  get_press_releases: "Fetching press releases",
  get_market_news: "Fetching market news",
  get_sector_peers: "Finding sector peers",
}

interface StockExplainChatProps {
  symbol: string
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function StockExplainChat({ symbol, open, onOpenChange }: StockExplainChatProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [toolCallStatus, setToolCallStatus] = useState<ToolCallStatus | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Start explanation automatically when opened
  useEffect(() => {
    if (open && messages.length === 0) {
      sendMessage(
        "Please analyze this stock and explain whether it's a good investment opportunity based on the pipeline analysis."
      )
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // Focus input when drawer opens
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [open])

  // Reset state when drawer closes
  useEffect(() => {
    if (!open) {
      setMessages([])
      setInput("")
      setError(null)
      setToolCallStatus(null)
    }
  }, [open])

  const sendMessage = async (userMessage: string) => {
    if (!userMessage.trim() || isLoading) return

    setError(null)
    const newMessages: Message[] = [...messages, { role: "user", content: userMessage }]
    setMessages(newMessages)
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch(`http://localhost:8000/api/v1/explain/${symbol}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: newMessages.map((m) => ({ role: m.role, content: m.content })),
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP error: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error("No response body")

      let assistantMessage = ""

      // Add empty assistant message for streaming
      setMessages([...newMessages, { role: "assistant", content: "" }])

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6)
            if (data === "[DONE]") continue

            try {
              const parsed = JSON.parse(data)

              // Handle tool call status
              if (parsed.tool_call) {
                setToolCallStatus({
                  name: parsed.tool_call,
                  args: parsed.args || {},
                })
                continue // Don't add to message yet
              }

              // Handle content - clear tool status when we get content
              if (parsed.content) {
                setToolCallStatus(null)
                assistantMessage += parsed.content
                setMessages((msgs) => {
                  const updated = [...msgs]
                  if (updated.length > 0) {
                    updated[updated.length - 1] = {
                      role: "assistant",
                      content: assistantMessage,
                    }
                  }
                  return updated
                })
              }
            } catch {
              // Ignore parse errors for incomplete chunks
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
      // Remove the empty assistant message if there was an error
      setMessages(newMessages)
    } finally {
      setIsLoading(false)
      setToolCallStatus(null)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(input)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  // Filter out the initial system-like prompt from display
  const displayMessages = messages.filter(
    (m) => m.role === "assistant" || !m.content.startsWith("Please analyze this stock")
  )

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent className="flex flex-col sm:max-w-xl">
        <DrawerHeader>
          <div className="flex items-center gap-2">
            <RiSparklingLine className="size-5 text-indigo-600 dark:text-indigo-400" />
            <DrawerTitle>AI Analysis: {symbol}</DrawerTitle>
          </div>
          <DrawerDescription>
            AI-powered explanation based on the 4-stage pipeline analysis
          </DrawerDescription>
        </DrawerHeader>

        <DrawerBody className="flex flex-col overflow-hidden p-0">
          {/* Messages area */}
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
            {displayMessages.map((message, index) => (
              <div
                key={index}
                className={cx("flex", message.role === "user" ? "justify-end" : "justify-start")}
              >
                <div
                  className={cx(
                    "max-w-[85%] rounded-lg px-4 py-2.5",
                    message.role === "user"
                      ? "bg-indigo-600 text-white"
                      : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700"
                  )}
                >
                  {message.content ? (
                    message.role === "assistant" ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-gray-900 dark:prose-headings:text-gray-100 prose-p:text-gray-700 dark:prose-p:text-gray-300 prose-strong:text-gray-900 dark:prose-strong:text-gray-100 prose-li:text-gray-700 dark:prose-li:text-gray-300 prose-headings:mt-3 prose-headings:mb-2 prose-p:my-1.5 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <div className="text-sm leading-relaxed">
                        {message.content}
                      </div>
                    )
                  ) : toolCallStatus ? (
                    <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400">
                      <RiLoader2Fill className="size-4 animate-spin" />
                      <span className="text-sm">
                        {TOOL_DISPLAY_NAMES[toolCallStatus.name] || toolCallStatus.name}
                        {toolCallStatus.args.symbol ? ` for ${String(toolCallStatus.args.symbol)}` : ""}
                        {toolCallStatus.args.sector ? ` in ${String(toolCallStatus.args.sector)}` : ""}
                        ...
                      </span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
                      <RiLoader2Fill className="size-4 animate-spin" />
                      <span className="text-sm">Analyzing...</span>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-3 dark:border-red-800 dark:bg-red-950">
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </DrawerBody>

        <DrawerFooter className="flex-col gap-3 sm:flex-col">
          <form onSubmit={handleSubmit} className="flex w-full gap-2">
            <Input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a follow-up question..."
              disabled={isLoading}
              className="flex-1"
            />
            <Button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="shrink-0"
            >
              <RiSendPlaneFill className="size-4" />
            </Button>
          </form>
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
            Ask about news, press releases, or competitors for live data
          </p>
        </DrawerFooter>
      </DrawerContent>
    </Drawer>
  )
}
