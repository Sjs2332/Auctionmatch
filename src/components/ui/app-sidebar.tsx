"use client";

import * as React from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { Database, Github } from "lucide-react";

const NAV_ITEMS = [
  {
    id: "analyzer",
    title: "Inventory Matcher",
    description: "Deep data insights",
    icon: Database,
  },
];

export function AppSidebar() {
  return (
    <Sidebar collapsible="none" className="h-svh sticky top-0 border-r border-slate-200 bg-[#fafafa] text-slate-800">
      <SidebarHeader className="h-16 px-6 flex flex-col justify-center border-b border-slate-200">
        <div className="flex flex-col leading-tight">
          <p className="text-sm font-black text-slate-900 uppercase tracking-tighter">AuctionMatch</p>
          <p className="text-[10px] text-slate-500 font-bold leading-tight mt-0.5 opacity-70 italic">Vectorized Analytical Core</p>
        </div>
      </SidebarHeader>
      <SidebarContent className="py-4">
        <SidebarGroup>
          <SidebarGroupLabel className="px-3 text-[10px] font-bold uppercase tracking-widest text-slate-400">
            Engine Terminal
          </SidebarGroupLabel>
          <SidebarGroupContent className="mt-4">
            <SidebarMenu>
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                return (
                  <SidebarMenuItem key={item.id}>
                    <SidebarMenuButton
                      type="button"
                      isActive={true}
                      className="flex items-center gap-3 rounded-none border-y border-x-0 border-slate-200 bg-white px-6 py-4 text-sm font-bold text-slate-900 transition hover:bg-slate-50"
                    >
                      <Icon className="h-4 w-4 text-sky-600" />
                      <span>{item.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="p-4 border-t border-slate-100">
        <a
          href="https://github.com/Sjs2332/Auctionmatch"
          target="_blank"
          rel="noopener noreferrer"
          className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs font-semibold text-slate-600 shadow-sm transition hover:bg-slate-50 hover:text-slate-900"
        >
          <Github className="h-3.5 w-3.5" />
          View Source Repository
        </a>
      </SidebarFooter>
    </Sidebar >
  );
}
